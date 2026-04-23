#!/usr/bin/env python3
"""
Orchestration script for MM-DBGDGM pipeline on high-spec VM.
Handles:
1. GDrive Download (Full folder download)
2. Parallel ZIP Extraction
3. Subject ID Matching (fMRI <-> sMRI)
4. Training with MI300X optimizations
5. Best Model & Results Export
"""

import os
import sys
import shutil
import zipfile
import logging
import argparse
import subprocess
import re
from pathlib import Path
from datetime import datetime
import concurrent.futures
from typing import List, Dict, Set, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mm-dbgdgm-pipeline")

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACT_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "best_model"

GDRIVE_FOLDER_ID = "1Qx3jqL_eSxGfWvhxb20M6orxvKsQlQT9"

# Common fMRI clue tokens from dataset.py
FMRI_CLUE_TOKENS = ("fmri", "bold", "rest", "rsfmri", "ep2d", "epi")

def ensure_dirs():
    for d in [RAW_DIR, EXTRACT_DIR, PROCESSED_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def download_from_gdrive():
    logger.info(f"Downloading data from Google Drive folder: {GDRIVE_FOLDER_ID}")
    try:
        import gdown
    except ImportError:
        logger.info("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    try:
        # download_folder is best for a shared folder link
        gdown.download_folder(id=GDRIVE_FOLDER_ID, output=str(RAW_DIR), quiet=False, use_cookies=False)
    except Exception as e:
        logger.error(f"gdown.download_folder failed: {e}")
        logger.info("Note: If the folder is very large or restricted, you may need to download manually to data/raw/")

def extract_zip(zip_path: Path, target_dir: Path):
    logger.info(f"Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    except Exception as e:
        logger.error(f"Failed to extract {zip_path.name}: {e}")

def get_subject_id(path: Path) -> str:
    """Extract subject ID from path using patterns from dataset.py."""
    # Pattern for ADNI (e.g. 002_S_0295)
    adni_match = re.search(r"(\d{3}_S_\d{4})", str(path))
    if adni_match:
        return adni_match.group(1)
    
    # Pattern for OASIS (e.g. OAS30001)
    oasis_match = re.search(r"(OAS\d+)", str(path))
    if oasis_match:
        return oasis_match.group(1)
    
    return ""

def is_fmri(path: Path) -> bool:
    """Check if a path likely contains fMRI data."""
    path_lower = str(path).lower()
    return any(token in path_lower for token in FMRI_CLUE_TOKENS)

def preprocess_and_match():
    logger.info("Preprocessing and matching subjects...")
    
    zip_files = list(RAW_DIR.glob("*.zip"))
    if not zip_files:
        logger.warning(f"No zip files found in {RAW_DIR}. Checking if data already exists in {EXTRACT_DIR}...")
        if not any(EXTRACT_DIR.iterdir()):
            logger.error("No data found to preprocess!")
            return False
    else:
        logger.info(f"Found {len(zip_files)} zip files. Extracting...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(lambda z: extract_zip(z, EXTRACT_DIR), zip_files)

    fmri_map: Dict[str, List[Path]] = {}
    smri_map: Dict[str, List[Path]] = {}

    logger.info("Scanning extracted files for subject modalities...")
    # Look for leaf-ish directories that contain data
    for root, dirs, files in os.walk(EXTRACT_DIR):
        root_path = Path(root)
        
        # Skip empty directories
        if not files and not dirs:
            continue
            
        subj_id = get_subject_id(root_path)
        if not subj_id:
            continue
            
        if is_fmri(root_path):
            fmri_map.setdefault(subj_id, []).append(root_path)
        else:
            # Assume it's sMRI if it has DICOMs/Images or is named sMRI
            smri_map.setdefault(subj_id, []).append(root_path)

    common_subjects = set(fmri_map.keys()) & set(smri_map.keys())
    logger.info(f"Total subjects: {len(set(fmri_map.keys()) | set(smri_map.keys()))}")
    logger.info(f"fMRI subjects: {len(fmri_map)}")
    logger.info(f"sMRI subjects: {len(smri_map)}")
    logger.info(f"Matched subjects: {len(common_subjects)}")

    if not common_subjects:
        logger.error("No subjects found with both fMRI and sMRI!")
        return False

    # Organize into PROCESSED_DIR
    logger.info(f"Moving {len(common_subjects)} matched subjects to {PROCESSED_DIR}")
    for subj_id in common_subjects:
        # For each subject, we might have multiple timepoints or folders. 
        # For simplicity, we'll take the first/best one.
        # Ideally, we should preserve the structure if it's already {subj_id}/{timepoint}
        
        # We'll just copy the directories into processed/fmri and processed/smri
        target_fmri = PROCESSED_DIR / "fmri" / subj_id
        target_smri = PROCESSED_DIR / "smri" / subj_id
        
        target_fmri.parent.mkdir(parents=True, exist_ok=True)
        target_smri.parent.mkdir(parents=True, exist_ok=True)
        
        # We use move to save space
        try:
            if not target_fmri.exists():
                shutil.move(str(fmri_map[subj_id][0]), str(target_fmri))
            if not target_smri.exists():
                shutil.move(str(smri_map[subj_id][0]), str(target_smri))
        except Exception as e:
            logger.warning(f"Could not move data for {subj_id}: {e}")

    # Generate labels.csv
    labels_csv = PROCESSED_DIR / "labels.csv"
    if not labels_csv.exists():
        logger.info("Creating placeholder labels.csv...")
        import pandas as pd
        data = []
        for subj_id in common_subjects:
            data.append({"subject_id": subj_id, "timepoint": "T0", "label": 0})
        pd.DataFrame(data).to_csv(labels_csv, index=False)
        logger.info(f"Created {labels_csv} with {len(data)} entries.")

    return True

def run_training():
    logger.info("Launching training...")
    
    # Results will go to a timestamped folder inside results/
    results_dir = ROOT_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "train_local.py",
        "--dataset-root", str(PROCESSED_DIR),
        "--metadata-file", str(PROCESSED_DIR / "labels.csv"),
        "--output-dir", str(results_dir),
        "--num-workers", "20",
        "--batch-size", "64",
        "--device", "cuda"
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run and stream output to terminal
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    
    if process.returncode != 0:
        logger.error("Training failed!")
        return False

    # Evaluation results and model handling
    best_model_pt = results_dir / "best_model.pt"
    if best_model_pt.exists():
        logger.info("Training successful! Storing best model and inference samples.")
        
        # Store only the best model in MODEL_DIR, delete old ones
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(str(best_model_pt), str(MODEL_DIR / "best_model.pt"))
        
        # Copy inference samples
        inf_samples = results_dir / "inference_samples"
        if inf_samples.exists():
            shutil.copytree(str(inf_samples), str(MODEL_DIR / "inference_samples"))
            
        # Copy test results
        test_results = results_dir / "test_results.json"
        if test_results.exists():
            shutil.copy(str(test_results), str(MODEL_DIR / "test_results.json"))
            
        logger.info(f"Final artifacts stored in {MODEL_DIR}")
    else:
        logger.error("Best model checkpoint not found in results!")
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(description="MM-DBGDGM Automated Pipeline")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    if not args.skip_download:
        download_from_gdrive()
    
    if not args.skip_preprocess:
        if not preprocess_and_match():
            logger.error("Matching phase failed. Aborting.")
            return

    if run_training():
        logger.info("Pipeline executed successfully.")
        # Cleanup extracted data to save space, but keep processed data
        if EXTRACT_DIR.exists():
            shutil.rmtree(EXTRACT_DIR)
            logger.info("Cleaned up temporary extraction directory.")
    else:
        logger.error("Pipeline failed during training.")

if __name__ == "__main__":
    main()
