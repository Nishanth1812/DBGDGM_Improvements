#!/usr/bin/env python3
"""Tiny demo harness that creates a fake dataset and runs prepare_wekan_data.py."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
except Exception:
    pydicom = None


def _write_fake_dicom(path: Path, seed: int) -> None:
    if pydicom is None:
        path.write_bytes(b"FAKE_DICOM")
        return
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PatientName = "Demo"
    ds.Rows = 8
    ds.Columns = 8
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    rng = np.random.default_rng(seed)
    pixel_array = (rng.random((8, 8)) * 1024).astype(np.uint16)
    ds.PixelData = pixel_array.tobytes()

    ds.save_as(str(path), write_like_original=False)


def build_demo_dataset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    fmri_root = root / "fMRI"
    smri_root = root / "sMRI"
    fmri_root.mkdir(parents=True)
    smri_root.mkdir(parents=True)

    subjects = [("SUBJ001", "MCI"), ("SUBJ002", "AD")]

    for subject_id, label in subjects:
        fmri_sub = fmri_root / subject_id
        smri_sub = smri_root / subject_id
        fmri_sub.mkdir(parents=True)
        smri_sub.mkdir(parents=True)

        for i in range(3):
            _write_fake_dicom(fmri_sub / f"{subject_id}_fmri_{i}.dcm", seed=i)
            _write_fake_dicom(smri_sub / f"{subject_id}_smri_{i}.dcm", seed=i + 10)

    (fmri_root / "labels.csv").write_text(
        "subject_id,class\nSUBJ001,MCI\nSUBJ002,AD\n",
        encoding="utf-8",
    )
    (smri_root / "labels.csv").write_text(
        "subject_id,class\nSUBJ001,MCI\nSUBJ002,AD\n",
        encoding="utf-8",
    )


def main() -> int:
    demo_root = Path("tools/wekan_prep/demo_data").resolve()
    build_demo_dataset(demo_root)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "prepare_wekan_data.py"),
        "--input-root",
        str(demo_root),
        "--fmri-root",
        str(demo_root / "fMRI"),
        "--smri-root",
        str(demo_root / "sMRI"),
        "--accept-any-files",
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

