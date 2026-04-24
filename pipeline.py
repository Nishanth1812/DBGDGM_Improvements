#!/usr/bin/env python3
"""
MM-DBGDGM End-to-End Pipeline
─────────────────────────────
Stages:
  1. [Download]    Pull raw data from Google Drive
  2. [Extract]     Unzip into data/extracted/
  3. [Pair]        Match fMRI <-> sMRI by ADNI subject ID
  4. [Label]       Infer CN/eMCI/lMCI/AD from folder path
  5. [Features]    Precompute sMRI proxy features (features.npy)
  6. [CSV]         Write data/processed/labels.csv
  7. [Train]       Launch train_local.py
  8. [Export]      Archive best model to best_model/

Flags:
  --skip-download    Use existing data/raw/  (skip GDrive)
  --skip-preprocess  Skip stages 2-6        (labels.csv must exist)
  --reprocess        Re-run stages 2-6 only (keeps data/extracted/)
  --force            Full reset: delete extracted + processed + model
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
from collections import Counter
import concurrent.futures

# ═══════════════════════════════════════════════════════════════════════════════
# Logging — two handlers: coloured console + plain file
# ═══════════════════════════════════════════════════════════════════════════════
LOG_FILE = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class _StageFormatter(logging.Formatter):
    LEVEL_COLOURS = {
        logging.DEBUG:    "\033[37m",   # grey
        logging.INFO:     "\033[36m",   # cyan
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLOURS.get(record.levelno, "")
        ts = datetime.now().strftime("%H:%M:%S")
        level = f"{colour}{record.levelname:<8}{self.RESET}"
        return f"[{ts}] {level} {record.getMessage()}"


class _PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{ts}] [{record.levelname}] {record.getMessage()}"


_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_StageFormatter())

_file_handler = logging.FileHandler(LOG_FILE)
_file_handler.setFormatter(_PlainFormatter())

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger("pipeline")


def _banner(title: str) -> None:
    """Print a prominent stage banner."""
    width = 60
    line = "═" * width
    logger.info(line)
    logger.info(f"  {title}")
    logger.info(line)


def _section(title: str) -> None:
    logger.info(f"── {title} {'─' * max(0, 55 - len(title))}")


# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
ROOT_DIR      = Path(__file__).resolve().parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
EXTRACT_DIR   = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR     = ROOT_DIR / "best_model"

GDRIVE_FOLDER_ID = "1Qx3jqL_eSxGfWvhxb20M6orxvKsQlQT9"

LABEL_NAMES = {0: "CN", 1: "eMCI", 2: "lMCI", 3: "AD"}
BATCH_SIZE  = 16   # ← reduced for better gradient resolution on small dataset

# ── Modality tokens ───────────────────────────────────────────────────────────
FMRI_TOKENS = ("fmri", "bold", "rest", "rsfmri", "ep2d", "epi",
                "functional", "asl", "pcasl", "resting")
SMRI_TOKENS = ("mprage", "t1", "structural", "spgr", "t1w",
                "bravo", "axial", "sagittal", "coronal")

# ── Diagnosis label inference (checked in priority order) ────────────────────
# Matches any part of the FULL folder path (case-insensitive).
# Add/extend keywords to match your exact ADNI folder naming.
LABEL_RULES = [
    (3, ["alzheimer", "alzheimers", "/ad/",  "_ad_",  "\\ad\\",  "dementia"]),
    (2, ["lmci", "late_mci",  "late-mci",  "late mci"]),
    (1, ["emci", "early_mci", "early-mci", "early mci", "/mci/", "_mci_", "\\mci\\"]),
    (0, ["cognitively_normal", "cogn_normal", "normal_control", "/cn/", "_cn_", "\\cn\\"]),
]


def ensure_dirs() -> None:
    for d in [RAW_DIR, EXTRACT_DIR, PROCESSED_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Download
# ═══════════════════════════════════════════════════════════════════════════════
def stage_download() -> None:
    _banner("STAGE 1 / 7 — Download from Google Drive")
    logger.info(f"GDrive folder ID : {GDRIVE_FOLDER_ID}")
    logger.info(f"Destination      : {RAW_DIR}")
    try:
        import gdown
    except ImportError:
        logger.info("gdown not found — installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
    try:
        gdown.download_folder(id=GDRIVE_FOLDER_ID, output=str(RAW_DIR),
                              quiet=False, use_cookies=False)
        logger.info("Download complete.")
    except Exception as exc:
        logger.error(f"Download failed: {exc}")
        logger.warning("Place files manually in data/raw/ and re-run with --skip-download")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Extract ZIPs
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_one(zip_path: Path, target: Path) -> None:
    logger.info(f"  Extracting {zip_path.name} ...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target)
        logger.info(f"  ✓ {zip_path.name}")
    except Exception as exc:
        logger.error(f"  ✗ {zip_path.name}: {exc}")


def stage_extract() -> None:
    _banner("STAGE 2 / 7 — Extract ZIP archives")
    zip_files = list(RAW_DIR.glob("*.zip"))
    if not zip_files:
        logger.warning(f"No .zip files found in {RAW_DIR} — skipping extraction.")
        return
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        logger.info("data/extracted/ is non-empty — skipping re-extraction.")
        logger.info("  (run with --force to re-extract from scratch)")
        return
    logger.info(f"Found {len(zip_files)} zip file(s) — extracting with 4 threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pool.map(lambda z: _extract_one(z, EXTRACT_DIR), zip_files)
    logger.info("Extraction complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Scan & Pair
# ═══════════════════════════════════════════════════════════════════════════════
def _get_subject_id(path: Path) -> str:
    match = re.search(r"(\d{3})[_.\-]S[_.\-](\d{4})", str(path))
    return f"{match.group(1)}_S_{match.group(2)}" if match else ""


def _infer_label(path: Path) -> int:
    """Return diagnosis label from full folder path. Defaults to 0 (CN/unknown)."""
    p = str(path).lower().replace("\\", "/")
    for label_int, keywords in LABEL_RULES:
        if any(kw in p for kw in keywords):
            return label_int
    return 0


def stage_scan_and_pair() -> dict:
    """
    Walk EXTRACT_DIR, classify every leaf folder as fMRI or sMRI,
    group by subject ID, intersect, and return paired dict.
    Returns: { subj_id: {'fmri': Path, 'smri': Path, 'label': int} }
    """
    _banner("STAGE 3 / 7 — Scan & pair fMRI ↔ sMRI subjects")
    fmri_map: dict = {}
    smri_map: dict = {}
    label_map: dict = {}
    folder_count = 0

    logger.info(f"Walking {EXTRACT_DIR} ...")
    for root, _dirs, files in os.walk(EXTRACT_DIR):
        if not files:
            continue
        root_path = Path(root)
        sid = _get_subject_id(root_path)
        if not sid:
            continue

        folder_count += 1
        p = str(root_path).lower()
        is_f = any(t in p for t in FMRI_TOKENS)
        is_s = any(t in p for t in SMRI_TOKENS)

        if   is_f and not is_s:     fmri_map.setdefault(sid, []).append(root_path)
        elif is_s and not is_f:     smri_map.setdefault(sid, []).append(root_path)
        elif is_f and is_s:         fmri_map.setdefault(sid, []).append(root_path)
        elif len(files) > 50:       fmri_map.setdefault(sid, []).append(root_path)
        else:                       smri_map.setdefault(sid, []).append(root_path)

        if sid not in label_map:
            label_map[sid] = _infer_label(root_path)

    paired_ids = sorted(set(fmri_map) & set(smri_map))
    fmri_only  = set(fmri_map) - set(smri_map)
    smri_only  = set(smri_map) - set(fmri_map)

    _section("Scan results")
    logger.info(f"  Leaf folders scanned      : {folder_count}")
    logger.info(f"  Unique subjects (fMRI)    : {len(fmri_map)}")
    logger.info(f"  Unique subjects (sMRI)    : {len(smri_map)}")
    logger.info(f"  Paired (both modalities)  : {len(paired_ids)}")
    if fmri_only:
        logger.warning(f"  fMRI-only (no sMRI)       : {len(fmri_only)}  e.g. {list(fmri_only)[:3]}")
    if smri_only:
        logger.warning(f"  sMRI-only (no fMRI)       : {len(smri_only)}  e.g. {list(smri_only)[:3]}")

    if not paired_ids:
        logger.error("No paired subjects found. Check your EXTRACT_DIR folder structure.")
        return {}

    paired = {}
    label_counts = Counter()
    for sid in paired_ids:
        lbl = label_map.get(sid, 0)
        label_counts[lbl] += 1
        paired[sid] = {'fmri': fmri_map[sid][0], 'smri': smri_map[sid][0], 'label': lbl}

    _section("Label distribution (from folder paths)")
    for lbl in sorted(LABEL_NAMES):
        bar  = "█" * label_counts[lbl]
        pct  = 100 * label_counts[lbl] / max(len(paired_ids), 1)
        logger.info(f"  {LABEL_NAMES[lbl]:6s} [{lbl}]: {label_counts[lbl]:3d} ({pct:5.1f}%)  {bar}")

    if label_counts[0] == len(paired_ids):
        logger.warning(
            "ALL subjects labelled CN (0). This usually means the diagnosis class "
            "is not in the folder path. Check your EXTRACT_DIR structure and extend "
            "LABEL_RULES in pipeline.py to match your folder names."
        )

    return paired


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4-6 — Organise, compute features, write CSV
# ═══════════════════════════════════════════════════════════════════════════════
def stage_preprocess(paired: dict) -> bool:
    _banner("STAGE 4-6 / 7 — Organise data + compute features + write CSV")

    import numpy as np
    import pandas as pd
    from MM_DBGDGM.data.dataset import _load_image_folder_proxy_features

    # ── 4: Copy matched folders into PROCESSED_DIR ───────────────────────────
    _section("Organising subject folders")
    copied = 0
    for sid, info in paired.items():
        for modality, src in [("fmri", info['fmri']), ("smri", info['smri'])]:
            dst = PROCESSED_DIR / modality / sid
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    shutil.copytree(str(src), str(dst))
                    copied += 1
                except Exception as exc:
                    logger.warning(f"  Could not copy {sid}/{modality}: {exc}")
    logger.info(f"  Copied {copied} new folder(s) into {PROCESSED_DIR}")

    # ── 5: Precompute sMRI features ──────────────────────────────────────────
    _section("Precomputing sMRI proxy features")
    rows = []
    skipped = 0
    n = len(paired)

    for i, (sid, info) in enumerate(sorted(paired.items()), 1):
        fmri_path = (PROCESSED_DIR / "fmri" / sid).resolve()
        smri_path = (PROCESSED_DIR / "smri" / sid).resolve()

        if not fmri_path.exists() or not smri_path.exists():
            logger.warning(f"  [{i:3d}/{n}] {sid} — processed folder missing, skipping.")
            skipped += 1
            continue

        features_file = smri_path / "features.npy"
        if not features_file.exists():
            feats = _load_image_folder_proxy_features(smri_path)
            if feats is None:
                logger.warning(f"  [{i:3d}/{n}] {sid} — sMRI feature extraction failed, skipping.")
                skipped += 1
                continue
            np.save(str(features_file), feats)
            logger.info(f"  [{i:3d}/{n}] {sid} — features.npy saved ({feats.shape})")
        else:
            logger.info(f"  [{i:3d}/{n}] {sid} — features.npy already exists, reusing.")

        fmri_npy = fmri_path / "fmri.npy"
        fmri_ref = str(fmri_npy) if fmri_npy.exists() else str(fmri_path)

        rows.append({
            "subject_id": sid,
            "timepoint":  "T0",
            "label":      info['label'],
            "fmri_path":  fmri_ref,
            "smri_path":  str(features_file),
        })

    if not rows:
        logger.error("No valid subjects after feature computation. Aborting.")
        return False

    # ── 6: Write labels.csv ──────────────────────────────────────────────────
    _section("Writing labels.csv")
    labels_csv = PROCESSED_DIR / "labels.csv"
    df = pd.DataFrame(rows)
    df.to_csv(labels_csv, index=False)

    label_dist = Counter(df['label'])
    logger.info(f"  Saved: {labels_csv}")
    logger.info(f"  Total subjects : {len(df)}")
    logger.info(f"  Skipped        : {skipped}")
    for lbl in sorted(LABEL_NAMES):
        logger.info(f"    {LABEL_NAMES[lbl]:6s} [{lbl}]: {label_dist[lbl]}")

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 7 — Training
# ═══════════════════════════════════════════════════════════════════════════════
def stage_train() -> bool:
    _banner("STAGE 7 / 7 — Training")

    labels_csv = PROCESSED_DIR / "labels.csv"
    if not labels_csv.exists():
        logger.error(f"labels.csv not found: {labels_csv}")
        return False

    results_dir = ROOT_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "train_local.py",
        "--dataset-root",  str(PROCESSED_DIR),
        "--metadata-file", str(labels_csv),
        "--output-dir",    str(results_dir),
        "--num-workers",   "4",
        "--batch-size",    str(BATCH_SIZE),
    ]
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Batch size: {BATCH_SIZE}")

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()

    if process.returncode != 0:
        logger.error(f"Training exited with code {process.returncode}.")
        return False

    # ── Export artefacts ─────────────────────────────────────────────────────
    best_pt = results_dir / "best_model.pt"
    if not best_pt.exists():
        logger.error("best_model.pt not produced. Training may have failed silently.")
        return False

    logger.info("Training successful — archiving artefacts...")
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(best_pt), str(MODEL_DIR / "best_model.pt"))

    for name in ("inference_samples", "test_results.json",
                 "training_history.json", "run_summary.json"):
        src = results_dir / name
        if src.exists():
            dst = MODEL_DIR / name
            shutil.copytree(str(src), str(dst)) if src.is_dir() else shutil.copy(str(src), str(dst))

    logger.info(f"  Best model   : {MODEL_DIR / 'best_model.pt'}")
    logger.info(f"  Full results : {results_dir}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="MM-DBGDGM End-to-End Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-download",   action="store_true",
                        help="Skip GDrive download; use existing data/raw/")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip all preprocessing stages; labels.csv must exist")
    parser.add_argument("--reprocess",       action="store_true",
                        help="Re-run preprocessing only (delete data/processed/, keep data/extracted/)")
    parser.add_argument("--force",           action="store_true",
                        help="Full reset: delete data/extracted/, data/processed/, best_model/")
    args = parser.parse_args()

    start_time = datetime.now()
    _banner(f"MM-DBGDGM Pipeline  —  started {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {LOG_FILE}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if args.force:
        logger.info("--force: removing data/extracted/, data/processed/, best_model/")
        for d in [EXTRACT_DIR, PROCESSED_DIR, MODEL_DIR]:
            shutil.rmtree(d, ignore_errors=True)
            logger.info(f"  Removed {d}")
    elif args.reprocess:
        logger.info("--reprocess: removing data/processed/ only")
        shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
        logger.info(f"  Removed {PROCESSED_DIR}")

    ensure_dirs()

    # ── Run stages ────────────────────────────────────────────────────────────
    if not args.skip_download:
        stage_download()

    if not args.skip_preprocess:
        stage_extract()

        paired = stage_scan_and_pair()
        if not paired:
            logger.error("Pairing failed — aborting.")
            sys.exit(1)

        if not stage_preprocess(paired):
            logger.error("Preprocessing failed — aborting.")
            sys.exit(1)

    # ── Training ──────────────────────────────────────────────────────────────
    success = stage_train()

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = datetime.now() - start_time
    _banner("PIPELINE COMPLETE" if success else "PIPELINE FAILED")
    logger.info(f"  Status  : {'✓ SUCCESS' if success else '✗ FAILED'}")
    logger.info(f"  Elapsed : {str(elapsed).split('.')[0]}")
    logger.info(f"  Log     : {LOG_FILE}")
    if success:
        logger.info(f"  Model   : {MODEL_DIR / 'best_model.pt'}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
