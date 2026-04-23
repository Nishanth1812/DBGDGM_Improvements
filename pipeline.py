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
    """Extract ADNI subject ID (NNN_S_NNNN) from a path."""
    adni_match = re.search(r"(\d{3}_S_\d{4})", str(path))
    if adni_match:
        return adni_match.group(1)
    return ""

def is_fmri(path: Path) -> bool:
    """Check if a path likely contains fMRI data."""
    path_lower = str(path).lower()
    return any(token in path_lower for token in FMRI_CLUE_TOKENS)

def preprocess_and_match():
    logger.info("Preprocessing and matching subjects...")
    
    # Check if PROCESSED_DIR already has matched data AND a valid labels.csv with explicit paths
    labels_csv = PROCESSED_DIR / "labels.csv"
    if (PROCESSED_DIR / "fmri").exists() and (PROCESSED_DIR / "smri").exists() and labels_csv.exists():
        import pandas as pd
        try:
            df = pd.read_csv(labels_csv)
            if 'fmri_path' in df.columns and 'smri_path' in df.columns and len(df) > 0:
                logger.info(f"Valid labels.csv with explicit paths exists ({len(df)} subjects). Skipping extraction and matching.")
                return True
            else:
                logger.info("labels.csv exists but lacks explicit paths. Regenerating...")
                labels_csv.unlink()
        except Exception:
            labels_csv.unlink(missing_ok=True)

    zip_files = list(RAW_DIR.glob("*.zip"))
    if not zip_files:
        logger.warning(f"No zip files found in {RAW_DIR}. Checking if data already exists in {EXTRACT_DIR}...")
        if not EXTRACT_DIR.exists() or not any(EXTRACT_DIR.iterdir()):
            logger.error("No data found to preprocess!")
            return False
    else:
        # Check if we already have extracted content to avoid re-extracting
        if any(EXTRACT_DIR.iterdir()):
            logger.info(f"Extraction directory {EXTRACT_DIR} is not empty. Skipping re-extraction of zips.")
        else:
            logger.info(f"Found {len(zip_files)} zip files. Extracting...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(lambda z: extract_zip(z, EXTRACT_DIR), zip_files)

    fmri_map: Dict[str, List[Path]] = {}
    smri_map: Dict[str, List[Path]] = {}

    # Look for leaf directories that actually contain files
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if not files:
            continue
            
        root_path = Path(root)
        subj_id = get_subject_id(root_path)
        if not subj_id:
            continue
            
        # We prefer "leaf" directories that contain actual data files (DICOMs, npy, etc)
        if is_fmri(root_path):
            # If we already have a path for this subject, prefer deeper ones or just keep them
            fmri_map.setdefault(subj_id, []).append(root_path)
        else:
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
        
        # We use copy to keep the extracted folder intact as requested
        try:
            if not target_fmri.exists():
                shutil.copytree(str(fmri_map[subj_id][0]), str(target_fmri))
            if not target_smri.exists():
                shutil.copytree(str(smri_map[subj_id][0]), str(target_smri))
        except Exception as e:
            logger.warning(f"Could not copy data for {subj_id}: {e}")

    # Generate labels.csv with EXPLICIT absolute paths
    labels_csv = PROCESSED_DIR / "labels.csv"
    logger.info("Creating labels.csv with explicit modality paths...")
    import pandas as pd
    data = []
    for subj_id in common_subjects:
        # Use absolute paths to be 100% sure the loader finds them
        fmri_path = (PROCESSED_DIR / "fmri" / subj_id).resolve()
        smri_path = (PROCESSED_DIR / "smri" / subj_id).resolve()
        
        if fmri_path.exists() and smri_path.exists():
            data.append({
                "subject_id": subj_id,
                "timepoint": "T0",
                "label": 0,
                "fmri_path": str(fmri_path),
                "smri_path": str(smri_path),
            })
    
    if not data:
        logger.error("No valid subject folders found in data/processed! Matching might have failed.")
        return False
        
    pd.DataFrame(data).to_csv(labels_csv, index=False)
    logger.info(f"Created {labels_csv} with {len(data)} entries (explicit absolute paths).")

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
    parser.add_argument("--force", action="store_true", help="Force re-extraction and matching by clearing old data")
    args = parser.parse_args()

    if args.force:
        logger.info("Force flag detected. Clearing extracted and processed directories...")
        if EXTRACT_DIR.exists():
            shutil.rmtree(EXTRACT_DIR)
        if PROCESSED_DIR.exists():
            shutil.rmtree(PROCESSED_DIR)
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)

    ensure_dirs()

    if not args.skip_download:
        download_from_gdrive()
    
    if not args.skip_preprocess:
        if not preprocess_and_match():
            logger.error("Matching phase failed. Aborting.")
            return

    if run_training():
        logger.info("Pipeline executed successfully.")
        # We no longer automatically delete EXTRACT_DIR as requested by the user.
        # This allows for manual inspection and avoids re-extraction on subsequent runs.
    else:
        logger.error("Pipeline failed during training.")

if __name__ == "__main__":
    main()
