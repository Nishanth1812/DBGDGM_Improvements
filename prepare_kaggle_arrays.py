from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_FMRI_FILE_NAMES = (
    'fmri_windows_dbgdgm.npy',
    'timeseries_windows.npy',
    'windows.npy',
)


@dataclass
class BuildSummary:
    total_rows: int
    built_subject_rows: int
    skipped_rows: int
    total_window_samples: int
    fmri_shape: list[int]
    smri_shape: list[int]
    labels_shape: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build fmri.npy, smri.npy, and labels.npy for kaggle_mmdbgdgm.py from preprocessed outputs.'
    )
    parser.add_argument('--dataset-root', required=True, help='Preprocessed dataset root containing fmri/ and smri/ folders')
    parser.add_argument('--labels-csv', required=True, help='CSV with subject_id,label and optional timepoint columns')
    parser.add_argument('--output-dir', required=True, help='Directory where fmri.npy, smri.npy, and labels.npy will be written')
    parser.add_argument(
        '--smri-feature-count',
        type=int,
        default=None,
        help='Optional expected sMRI feature count. Use 5 for the current Kaggle trainer.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail immediately when a metadata row cannot be resolved instead of skipping it.',
    )
    return parser.parse_args()


def load_label_rows(labels_csv: Path) -> list[dict[str, str]]:
    with labels_csv.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError('Labels CSV is empty')

        required = {'subject_id', 'label'}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Labels CSV is missing required columns: {sorted(missing)}")

        rows = []
        for row in reader:
            subject_id = str(row['subject_id']).strip()
            if not subject_id:
                continue

            timepoint_raw = row.get('timepoint', '')
            timepoint = str(timepoint_raw).strip() if timepoint_raw is not None else ''
            rows.append(
                {
                    'subject_id': subject_id,
                    'timepoint': timepoint,
                    'label': str(row['label']).strip(),
                }
            )

    if not rows:
        raise ValueError('Labels CSV did not contain any usable rows')
    return rows


def resolve_file(dataset_root: Path, modality: str, subject_id: str, timepoint: str, file_names: Iterable[str]) -> Path | None:
    candidates = []
    if timepoint:
        candidates.extend(dataset_root / modality / subject_id / timepoint / file_name for file_name in file_names)
    candidates.extend(dataset_root / modality / subject_id / file_name for file_name in file_names)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_fmri_windows(file_path: Path) -> np.ndarray:
    fmri = np.load(file_path).astype(np.float32)
    if fmri.ndim != 3 or fmri.shape[1] != 200 or fmri.shape[2] != 50:
        raise ValueError(f'Expected fMRI windows shaped [N, 200, 50], got {tuple(fmri.shape)} from {file_path}')
    if fmri.shape[0] == 0:
        raise ValueError(f'fMRI file contains zero windows: {file_path}')
    return fmri


def load_smri_features(file_path: Path, expected_feature_count: int | None) -> np.ndarray:
    smri = np.load(file_path).astype(np.float32)

    if smri.ndim == 2 and smri.shape[0] == 1:
        smri = smri[0]
    elif smri.ndim != 1:
        raise ValueError(f'Expected sMRI features shaped [1, F] or [F], got {tuple(smri.shape)} from {file_path}')

    if expected_feature_count is not None and smri.shape[0] != expected_feature_count:
        raise ValueError(
            f'Expected {expected_feature_count} sMRI features, got {smri.shape[0]} from {file_path}'
        )

    return smri.astype(np.float32)


def build_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    dataset_root = Path(args.dataset_root)
    labels_csv = Path(args.labels_csv)

    rows = load_label_rows(labels_csv)
    fmri_batches: list[np.ndarray] = []
    smri_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, str]] = []

    for row in rows:
        subject_id = row['subject_id']
        timepoint = row['timepoint']
        label = int(row['label'])

        fmri_path = resolve_file(dataset_root, 'fmri', subject_id, timepoint, DEFAULT_FMRI_FILE_NAMES)
        smri_path = resolve_file(dataset_root, 'smri', subject_id, timepoint, ('features.npy',))

        if fmri_path is None or smri_path is None:
            reason = 'missing fMRI file' if fmri_path is None else 'missing sMRI file'
            skipped = {
                'subject_id': subject_id,
                'timepoint': timepoint,
                'reason': reason,
            }
            if args.strict:
                raise FileNotFoundError(str(skipped))
            skipped_rows.append(skipped)
            continue

        try:
            fmri = load_fmri_windows(fmri_path)
            smri = load_smri_features(smri_path, args.smri_feature_count)
        except Exception as exc:
            skipped = {
                'subject_id': subject_id,
                'timepoint': timepoint,
                'reason': str(exc),
            }
            if args.strict:
                raise
            skipped_rows.append(skipped)
            continue

        repeated_smri = np.repeat(smri[np.newaxis, :], fmri.shape[0], axis=0)
        repeated_labels = np.full(fmri.shape[0], label, dtype=np.int64)

        fmri_batches.append(fmri)
        smri_batches.append(repeated_smri)
        label_batches.append(repeated_labels)
        manifest_rows.append(
            {
                'subject_id': subject_id,
                'timepoint': timepoint,
                'label': label,
                'fmri_file': str(fmri_path),
                'smri_file': str(smri_path),
                'n_windows': int(fmri.shape[0]),
                'n_smri_features': int(smri.shape[0]),
            }
        )

    if not fmri_batches:
        raise ValueError('No training arrays were built. Check preprocessing outputs, labels CSV, and feature counts.')

    fmri_array = np.concatenate(fmri_batches, axis=0).astype(np.float32)
    smri_array = np.concatenate(smri_batches, axis=0).astype(np.float32)
    labels_array = np.concatenate(label_batches, axis=0).astype(np.int64)

    summary = {
        'build_summary': asdict(
            BuildSummary(
                total_rows=len(rows),
                built_subject_rows=len(manifest_rows),
                skipped_rows=len(skipped_rows),
                total_window_samples=int(len(labels_array)),
                fmri_shape=list(fmri_array.shape),
                smri_shape=list(smri_array.shape),
                labels_shape=list(labels_array.shape),
            )
        ),
        'skipped_rows': skipped_rows,
        'fmri_file_candidates': list(DEFAULT_FMRI_FILE_NAMES),
    }
    return fmri_array, smri_array, labels_array, {'summary': summary, 'manifest_rows': manifest_rows}


def write_outputs(output_dir: Path, fmri: np.ndarray, smri: np.ndarray, labels: np.ndarray, metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'fmri.npy', fmri)
    np.save(output_dir / 'smri.npy', smri)
    np.save(output_dir / 'labels.npy', labels)

    with (output_dir / 'build_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata['summary'], handle, indent=2)

    with (output_dir / 'samples_manifest.csv').open('w', encoding='utf-8', newline='') as handle:
        fieldnames = ['subject_id', 'timepoint', 'label', 'fmri_file', 'smri_file', 'n_windows', 'n_smri_features']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata['manifest_rows'])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fmri, smri, labels, metadata = build_arrays(args)
    write_outputs(output_dir, fmri, smri, labels, metadata)

    summary = metadata['summary']['build_summary']
    print('Built Kaggle arrays successfully')
    print(f"  fmri.npy: {tuple(summary['fmri_shape'])}")
    print(f"  smri.npy: {tuple(summary['smri_shape'])}")
    print(f"  labels.npy: {tuple(summary['labels_shape'])}")
    print(f"  subject rows used: {summary['built_subject_rows']}/{summary['total_rows']}")
    print(f"  skipped rows: {summary['skipped_rows']}")


if __name__ == '__main__':
    main()