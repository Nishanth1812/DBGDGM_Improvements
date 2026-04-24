#!/usr/bin/env python3
"""
MM-DBGDGM End-to-End Pipeline
Stages:
  1. Download from Google Drive (optional)
  2. Extract ZIP files into data/extracted/
  3. Scan extracted data, pair fMRI <-> sMRI by subject ID
  4. Infer diagnosis labels from folder path (CN/MCI/AD)
  5. Precompute sMRI proxy features (features.npy per subject)
  6. Write data/processed/labels.csv with explicit absolute paths
  7. Launch train_local.py → evaluate → save best model

Flags:
  --skip-download    Skip GDrive download (use existing data/raw/)
  --reprocess        Delete only data/processed/ and re-pair/re-label
                     (keeps data/extracted/ intact — use this flag normally)
  --force            Delete data/extracted/ AND data/processed/ — full reset
  --skip-preprocess  Skip straight to training (labels.csv must exist)
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

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mm-dbgdgm-pipeline")

ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
EXTRACT_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR  = ROOT_DIR / "best_model"

GDRIVE_FOLDER_ID = "1Qx3jqL_eSxGfWvhxb20M6orxvKsQlQT9"

# ── Modality classification tokens ──────────────────────────────────────────
FMRI_TOKENS = ("fmri", "bold", "rest", "rsfmri", "ep2d", "epi",
                "functional", "asl", "pcasl", "resting")
SMRI_TOKENS = ("mprage", "t1", "structural", "spgr", "t1w",
                "bravo", "axial", "sagittal", "coronal")

# ── Diagnosis label mapping from path keywords ───────────────────────────────
# ADNI stores diagnosis in folder names like:
#   extracted/Alzheimers_Disease/002_S_4213/...
#   extracted/CN/002_S_4213/...
#   extracted/MCI/002_S_4213/...
LABEL_KEYWORDS = [
    # (label_int, [path substrings to match, case-insensitive])
    (3, ["alzheimer", "alzheimers", "_ad_", "/ad/", "\\ad\\"]),
    (2, ["lmci", "late_mci", "late-mci"]),
    (1, ["emci", "early_mci", "early-mci", "mci"]),
    (0, ["cn", "cognitively_normal", "cogn_normal", "normal_control", "normal"]),
]


def ensure_dirs():
    for d in [RAW_DIR, EXTRACT_DIR, PROCESSED_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── Stage 1: Download ─────────────────────────────────────────────────────────
def download_from_gdrive():
    logger.info(f"Downloading data from Google Drive folder: {GDRIVE_FOLDER_ID}")
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    try:
        gdown.download_folder(id=GDRIVE_FOLDER_ID, output=str(RAW_DIR),
                              quiet=False, use_cookies=False)
    except Exception as e:
        logger.error(f"gdown failed: {e}")
        logger.info("Download manually to data/raw/ and re-run with --skip-download")


# ── Stage 2: Extract ──────────────────────────────────────────────────────────
def extract_zip(zip_path: Path, target_dir: Path):
    logger.info(f"Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
        logger.info(f"Done: {zip_path.name}")
    except Exception as e:
        logger.error(f"Failed to extract {zip_path.name}: {e}")


def extract_all():
    zip_files = list(RAW_DIR.glob("*.zip"))
    if not zip_files:
        logger.warning(f"No zip files in {RAW_DIR}.")
        return
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        logger.info("data/extracted/ is non-empty — skipping re-extraction.")
        return
    logger.info(f"Extracting {len(zip_files)} zip file(s)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        ex.map(lambda z: extract_zip(z, EXTRACT_DIR), zip_files)


# ── Stage 3: Subject ID extraction ────────────────────────────────────────────
def get_subject_id(path: Path) -> str:
    """Extract ADNI subject ID (NNN_S_NNNN) from a path."""
    match = re.search(r"(\d{3})[_.\-]S[_.\-](\d{4})", str(path))
    if match:
        return f"{match.group(1)}_S_{match.group(2)}"
    return ""


# ── Stage 4: Label inference from full path ────────────────────────────────────
def infer_label_from_path(path: Path) -> int:
    """
    Determine diagnosis label by scanning the full folder path.
    Priority: AD (3) > lMCI (2) > eMCI/MCI (1) > CN (0).
    Returns 0 (CN) if nothing matches — so it never crashes.
    """
    p = str(path).lower().replace("\\", "/")
    for label_int, keywords in LABEL_KEYWORDS:
        if any(kw in p for kw in keywords):
            return label_int
    return 0  # Default: CN / Unknown


# ── Stage 5: Scan and pair ────────────────────────────────────────────────────
def scan_and_pair():
    """
    Walk EXTRACT_DIR, classify every leaf folder as fMRI or sMRI,
    group by subject ID, and return only subjects that have both.
    Returns: dict { subj_id: {'fmri': Path, 'smri': Path, 'label': int} }
    """
    fmri_map: dict = {}  # subj_id -> [Path, ...]
    smri_map: dict = {}
    label_map: dict = {}  # subj_id -> int (from first encountered path)

    found_folders = 0
    logger.info(f"Scanning {EXTRACT_DIR} for ADNI subjects...")

    for root, dirs, files in os.walk(EXTRACT_DIR):
        if not files:
            continue
        root_path = Path(root)
        subj_id = get_subject_id(root_path)
        if not subj_id:
            continue

        found_folders += 1
        p_lower = str(root_path).lower()

        is_f = any(t in p_lower for t in FMRI_TOKENS)
        is_s = any(t in p_lower for t in SMRI_TOKENS)

        if is_f and not is_s:
            fmri_map.setdefault(subj_id, []).append(root_path)
        elif is_s and not is_f:
            smri_map.setdefault(subj_id, []).append(root_path)
        elif is_f and is_s:
            # Both tokens present — BOLD wins
            fmri_map.setdefault(subj_id, []).append(root_path)
        else:
            # Ambiguous: many files → fMRI (timeseries), few files → sMRI
            if len(files) > 50:
                fmri_map.setdefault(subj_id, []).append(root_path)
            else:
                smri_map.setdefault(subj_id, []).append(root_path)

        # Infer label from full path (done once per subject, from any folder)
        if subj_id not in label_map:
            label_map[subj_id] = infer_label_from_path(root_path)

    paired = sorted(set(fmri_map) & set(smri_map))

    logger.info("─── Scan Summary ───────────────────────────────────────")
    logger.info(f"  Leaf folders scanned : {found_folders}")
    logger.info(f"  Subjects with fMRI   : {len(fmri_map)}")
    logger.info(f"  Subjects with sMRI   : {len(smri_map)}")
    logger.info(f"  Paired subjects      : {len(paired)}")
    logger.info("────────────────────────────────────────────────────────")

    if not paired:
        logger.error("No paired subjects found. Check EXTRACT_DIR structure.")
        if fmri_map:
            logger.error(f"  Sample fMRI key: {list(fmri_map.keys())[0]}")
        if smri_map:
            logger.error(f"  Sample sMRI key: {list(smri_map.keys())[0]}")
        return {}

    # Count label distribution
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    result = {}
    for sid in paired:
        lbl = label_map.get(sid, 0)
        label_counts[lbl] += 1
        result[sid] = {
            'fmri': fmri_map[sid][0],
            'smri': smri_map[sid][0],
            'label': lbl,
        }

    logger.info(f"  Label distribution → CN:{label_counts[0]}  eMCI:{label_counts[1]}  lMCI:{label_counts[2]}  AD:{label_counts[3]}")
    return result


# ── Stage 6: Preprocess and write labels.csv ──────────────────────────────────
def preprocess_and_match() -> bool:
    import numpy as np
    import pandas as pd
    from MM_DBGDGM.data.dataset import _load_image_folder_proxy_features

    if not EXTRACT_DIR.exists() or not any(EXTRACT_DIR.rglob('*')):
        logger.error(f"EXTRACT_DIR is empty: {EXTRACT_DIR}")
        return False

    paired = scan_and_pair()
    if not paired:
        return False

    # Copy matched subjects into PROCESSED_DIR/fmri/ and PROCESSED_DIR/smri/
    logger.info(f"Organising {len(paired)} subjects into {PROCESSED_DIR}...")
    for subj_id, info in paired.items():
        target_fmri = PROCESSED_DIR / "fmri" / subj_id
        target_smri = PROCESSED_DIR / "smri" / subj_id
        target_fmri.parent.mkdir(parents=True, exist_ok=True)
        target_smri.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not target_fmri.exists():
                shutil.copytree(str(info['fmri']), str(target_fmri))
            if not target_smri.exists():
                shutil.copytree(str(info['smri']), str(target_smri))
        except Exception as e:
            logger.warning(f"Could not copy {subj_id}: {e}")

    # Precompute sMRI features and build labels.csv
    logger.info("Precomputing sMRI proxy features...")
    rows = []
    for i, (subj_id, info) in enumerate(sorted(paired.items())):
        fmri_path = (PROCESSED_DIR / "fmri" / subj_id).resolve()
        smri_path = (PROCESSED_DIR / "smri" / subj_id).resolve()

        if not fmri_path.exists() or not smri_path.exists():
            logger.warning(f"[{i+1}] Skipping {subj_id}: processed folder missing.")
            continue

        # sMRI features
        features_file = smri_path / "features.npy"
        if not features_file.exists():
            features = _load_image_folder_proxy_features(smri_path)
            if features is None:
                logger.warning(f"[{i+1}/{len(paired)}] Could not extract sMRI features for {subj_id}, skipping.")
                continue
            np.save(str(features_file), features)
            logger.info(f"[{i+1}/{len(paired)}] sMRI features saved → {features_file.name}")

        # fMRI reference: prefer precomputed .npy, otherwise point at folder
        fmri_npy = fmri_path / "fmri.npy"
        fmri_ref = str(fmri_npy) if fmri_npy.exists() else str(fmri_path)

        rows.append({
            "subject_id": subj_id,
            "timepoint":  "T0",
            "label":      info['label'],
            "fmri_path":  fmri_ref,
            "smri_path":  str(features_file),
        })

    if not rows:
        logger.error("No valid subjects after feature extraction. Aborting.")
        return False

    labels_csv = PROCESSED_DIR / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_csv, index=False)
    logger.info(f"labels.csv written: {labels_csv}  ({len(rows)} subjects)")

    # Print label distribution in the final CSV
    from collections import Counter
    label_dist = Counter(r['label'] for r in rows)
    logger.info(f"Final label distribution → CN:{label_dist[0]}  eMCI:{label_dist[1]}  lMCI:{label_dist[2]}  AD:{label_dist[3]}")
    return True


# ── Stage 7: Training ─────────────────────────────────────────────────────────
def run_training() -> bool:
    logger.info("Launching training...")
    results_dir = ROOT_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "train_local.py",
        "--dataset-root",  str(PROCESSED_DIR),
        "--metadata-file", str(PROCESSED_DIR / "labels.csv"),
        "--output-dir",    str(results_dir),
        "--num-workers",   "4",
        "--batch-size",    "32",
    ]
    logger.info(f"Command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()

    if process.returncode != 0:
        logger.error("Training failed!")
        return False

    best_model_pt = results_dir / "best_model.pt"
    if not best_model_pt.exists():
        logger.error("best_model.pt not found in results directory.")
        return False

    logger.info("Training successful! Archiving artefacts...")
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(best_model_pt), str(MODEL_DIR / "best_model.pt"))

    for extra in ["inference_samples", "test_results.json", "run_summary.json"]:
        src = results_dir / extra
        if src.exists():
            dst = MODEL_DIR / extra
            shutil.copytree(str(src), str(dst)) if src.is_dir() else shutil.copy(str(src), str(dst))

    logger.info(f"Artefacts stored in {MODEL_DIR}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MM-DBGDGM End-to-End Pipeline")
    parser.add_argument("--skip-download",   action="store_true",
                        help="Skip GDrive download (use existing data/raw/)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing — labels.csv must already exist")
    parser.add_argument("--reprocess",       action="store_true",
                        help="Delete data/processed/ and redo pairing/labelling "
                             "(keeps data/extracted/ intact)")
    parser.add_argument("--force",           action="store_true",
                        help="Full reset: delete data/extracted/ AND data/processed/")
    args = parser.parse_args()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if args.force:
        logger.info("--force: clearing data/extracted/ and data/processed/")
        shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
        shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
    elif args.reprocess:
        logger.info("--reprocess: clearing data/processed/ only (keeping data/extracted/)")
        shutil.rmtree(PROCESSED_DIR, ignore_errors=True)

    ensure_dirs()

    # ── Download ──────────────────────────────────────────────────────────────
    if not args.skip_download:
        download_from_gdrive()

    # ── Extract ───────────────────────────────────────────────────────────────
    if not args.skip_preprocess and not args.skip_download:
        extract_all()
    elif not args.skip_preprocess:
        extract_all()   # still extract even with --skip-download if needed

    # ── Preprocess & Match ────────────────────────────────────────────────────
    if not args.skip_preprocess:
        if not preprocess_and_match():
            logger.error("Preprocessing/matching failed. Aborting pipeline.")
            sys.exit(1)

    # ── Train ─────────────────────────────────────────────────────────────────
    if run_training():
        logger.info("Pipeline completed successfully.")
    else:
        logger.error("Pipeline failed during training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
