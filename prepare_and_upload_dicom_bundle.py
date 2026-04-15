"""Build a single upload bundle from messy DICOM downloads.

This script is meant for Google Drive exports like the folders in Downloads that
mix deep directory trees, loose DICOM files, and nested ZIP archives.

It recursively walks one or more input roots, extracts ZIP archives in-place
in memory, preserves the source folder hierarchy under a stable root alias, and
writes a single zip bundle that can be uploaded once to Modal's raw ZIP cache.

The MM-DBGDGM Modal trainer already extracts ZIPs from
`/data/adni_drive_pipeline/raw_zips` before preprocessing, so the bundle created
here can be uploaded once and then preprocessed during training.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import zipfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Set


DEFAULT_OUTPUT_ROOT = Path.home() / "Downloads" / "dicom_bundle"
DEFAULT_BUNDLE_NAME = "prepared_dicom_bundle"
DEFAULT_REMOTE_BASE_ROOT = "adni_drive_pipeline"
DEFAULT_VOLUME_NAME = "mm-dbgdgm-data"
DEFAULT_MAX_ARCHIVE_DEPTH = 8
DEFAULT_PROGRESS_EVERY = 1000
PYPICOM_UID_WARNING_PATTERN = r"Invalid value for VR UI:.*"

ALLOWED_DICOM_SUFFIXES = {"", ".dcm", ".dicom", ".ima"}
METADATA_FILENAMES = {"labels.csv"}
SKIP_PATH_PARTS = {"__macosx"}
SKIP_PREFIXES = ("._",)


@dataclass
class BundleStats:
    input_roots: int = 0
    filesystem_files_seen: int = 0
    archive_members_seen: int = 0
    dicom_files_written: int = 0
    metadata_files_written: int = 0
    archives_seen: int = 0
    archives_processed: int = 0
    archives_skipped: int = 0
    skipped_hidden: int = 0
    skipped_non_dicom: int = 0
    skipped_invalid_dicom: int = 0
    skipped_path_traversal: int = 0
    root_written_counts: Dict[str, int] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, object]:
        return {
            "input_roots": self.input_roots,
            "filesystem_files_seen": self.filesystem_files_seen,
            "archive_members_seen": self.archive_members_seen,
            "dicom_files_written": self.dicom_files_written,
            "metadata_files_written": self.metadata_files_written,
            "archives_seen": self.archives_seen,
            "archives_processed": self.archives_processed,
            "archives_skipped": self.archives_skipped,
            "skipped_hidden": self.skipped_hidden,
            "skipped_non_dicom": self.skipped_non_dicom,
            "skipped_invalid_dicom": self.skipped_invalid_dicom,
            "skipped_path_traversal": self.skipped_path_traversal,
            "root_written_counts": dict(sorted(self.root_written_counts.items())),
        }


@dataclass
class ProgressState:
    logger: logging.Logger
    progress_every: int = DEFAULT_PROGRESS_EVERY
    next_log_at: int = DEFAULT_PROGRESS_EVERY

    def maybe_log(self, stats: BundleStats, context: str) -> None:
        processed = stats.filesystem_files_seen + stats.archive_members_seen
        if processed < self.next_log_at:
            return

        self.logger.info(
            f"{context} | processed={processed} | filesystem_files={stats.filesystem_files_seen} | "
            f"archive_members={stats.archive_members_seen} | dicom_written={stats.dicom_files_written} | "
            f"metadata_written={stats.metadata_files_written} | skipped_non_dicom={stats.skipped_non_dicom} | "
            f"skipped_invalid_dicom={stats.skipped_invalid_dicom} | skipped_hidden={stats.skipped_hidden} | "
            f"skipped_traversal={stats.skipped_path_traversal}"
        )

        while self.next_log_at <= processed:
            self.next_log_at += self.progress_every


class _SuppressPydicomUidWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Invalid value for VR UI:" not in record.getMessage()


def _configure_pydicom_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=PYPICOM_UID_WARNING_PATTERN,
        category=UserWarning,
    )
    pydicom_logger = logging.getLogger("pydicom")
    if not any(isinstance(filter_obj, _SuppressPydicomUidWarningFilter) for filter_obj in pydicom_logger.filters):
        pydicom_logger.addFilter(_SuppressPydicomUidWarningFilter())


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("dicom-bundler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _normalize_bundle_component(name: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*]+", "_", name.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned or "root"


def _make_unique_alias(label: str, used_aliases: Set[str]) -> str:
    base = _normalize_bundle_component(label)
    alias = base
    counter = 2
    while alias in used_aliases:
        alias = f"{base}__{counter}"
        counter += 1
    used_aliases.add(alias)
    return alias


def _is_hidden_component(name: str) -> bool:
    lowered = name.lower()
    if lowered in {".ds_store", "thumbs.db"}:
        return True
    if lowered.startswith(SKIP_PREFIXES):
        return True
    if lowered in SKIP_PATH_PARTS:
        return True
    return False


def _path_has_hidden_part(path: Path) -> bool:
    return any(_is_hidden_component(part) for part in path.parts)


def _safe_archive_member_path(member_name: str) -> Optional[Path]:
    member_path = PurePosixPath(member_name)
    if member_path.is_absolute():
        return None

    safe_parts: List[str] = []
    for part in member_path.parts:
        if part in {".", ""}:
            continue
        if part == "..":
            return None
        safe_parts.append(part)

    if not safe_parts:
        return None

    return Path(*safe_parts)


def _candidate_dicom_suffix(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_DICOM_SUFFIXES


def _dataset_looks_like_dicom(dataset: object) -> bool:
    for attribute in ("SOPInstanceUID", "SeriesInstanceUID", "StudyInstanceUID", "Modality", "PatientID", "Rows", "Columns"):
        if hasattr(dataset, attribute):
            return True
    return False


def _is_valid_dicom_path(path: Path) -> bool:
    try:
        import pydicom

        with pydicom.config.disable_value_validation():
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return _dataset_looks_like_dicom(dataset)
    except Exception:
        return False


def _is_valid_dicom_bytes(payload: bytes) -> bool:
    try:
        import pydicom

        with pydicom.config.disable_value_validation():
            dataset = pydicom.dcmread(io.BytesIO(payload), stop_before_pixels=True, force=True)
        return _dataset_looks_like_dicom(dataset)
    except Exception:
        return False


def _record_written(stats: BundleStats, root_alias: str) -> None:
    stats.root_written_counts[root_alias] = stats.root_written_counts.get(root_alias, 0) + 1


def _write_metadata(zip_writer: zipfile.ZipFile, target_arcname: str, payload: bytes, stats: BundleStats, root_alias: str) -> None:
    zip_writer.writestr(target_arcname, payload)
    stats.metadata_files_written += 1
    _record_written(stats, root_alias)


def _write_dicom_path(zip_writer: zipfile.ZipFile, source_path: Path, target_arcname: str, stats: BundleStats, root_alias: str) -> None:
    zip_writer.write(source_path, target_arcname)
    stats.dicom_files_written += 1
    _record_written(stats, root_alias)


def _write_dicom_bytes(zip_writer: zipfile.ZipFile, target_arcname: str, payload: bytes, stats: BundleStats, root_alias: str) -> None:
    zip_writer.writestr(target_arcname, payload)
    stats.dicom_files_written += 1
    _record_written(stats, root_alias)


def _bundle_zip_archive(
    archive: zipfile.ZipFile,
    bundle_prefix: Path,
    bundle_zip: zipfile.ZipFile,
    stats: BundleStats,
    progress: ProgressState,
    root_alias: str,
    validate_dicom: bool,
    archive_depth: int,
    max_archive_depth: int,
) -> None:
    for member_info in archive.infolist():
        if member_info.is_dir():
            continue

        stats.archive_members_seen += 1

        safe_member_path = _safe_archive_member_path(member_info.filename)
        if safe_member_path is None:
            stats.skipped_path_traversal += 1
            continue
        if _path_has_hidden_part(safe_member_path):
            stats.skipped_hidden += 1
            continue

        member_bytes = archive.read(member_info)
        if not member_bytes:
            continue

        if zipfile.is_zipfile(io.BytesIO(member_bytes)):
            stats.archives_seen += 1
            if archive_depth >= max_archive_depth:
                stats.archives_skipped += 1
                continue

            nested_prefix = bundle_prefix / safe_member_path.parent / f"__zip__{safe_member_path.stem}"
            with zipfile.ZipFile(io.BytesIO(member_bytes), mode="r") as nested_archive:
                stats.archives_processed += 1
                _bundle_zip_archive(
                    archive=nested_archive,
                    bundle_prefix=nested_prefix,
                    bundle_zip=bundle_zip,
                    stats=stats,
                    progress=progress,
                    root_alias=root_alias,
                    validate_dicom=validate_dicom,
                    archive_depth=archive_depth + 1,
                    max_archive_depth=max_archive_depth,
                )
            progress.maybe_log(stats, f"{root_alias}: scanning nested archive {safe_member_path.name}")
            continue

        target_arcname = (bundle_prefix / safe_member_path).as_posix()
        if safe_member_path.name.lower() in METADATA_FILENAMES:
            _write_metadata(bundle_zip, target_arcname, member_bytes, stats, root_alias)
            continue

        if _candidate_dicom_suffix(safe_member_path.name):
            if (not validate_dicom) or _is_valid_dicom_bytes(member_bytes):
                _write_dicom_bytes(bundle_zip, target_arcname, member_bytes, stats, root_alias)
            else:
                stats.skipped_invalid_dicom += 1
            progress.maybe_log(stats, f"{root_alias}: scanning archive members")
            continue

        stats.skipped_non_dicom += 1
        progress.maybe_log(stats, f"{root_alias}: scanning archive members")


def _bundle_directory(
    source_dir: Path,
    bundle_prefix: Path,
    bundle_zip: zipfile.ZipFile,
    stats: BundleStats,
    progress: ProgressState,
    root_alias: str,
    validate_dicom: bool,
    archive_depth: int,
    max_archive_depth: int,
) -> None:
    for child_path in sorted(source_dir.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
        if _path_has_hidden_part(child_path):
            stats.skipped_hidden += 1
            continue

        if child_path.is_dir():
            _bundle_directory(
                source_dir=child_path,
                bundle_prefix=bundle_prefix / child_path.name,
                bundle_zip=bundle_zip,
                stats=stats,
                progress=progress,
                root_alias=root_alias,
                validate_dicom=validate_dicom,
                archive_depth=archive_depth,
                max_archive_depth=max_archive_depth,
            )
            continue

        stats.filesystem_files_seen += 1
        target_arcname = (bundle_prefix / child_path.name).as_posix()

        if child_path.name.lower() in METADATA_FILENAMES:
            _write_metadata(bundle_zip, target_arcname, child_path.read_bytes(), stats, root_alias)
            continue

        if zipfile.is_zipfile(child_path):
            stats.archives_seen += 1
            if archive_depth >= max_archive_depth:
                stats.archives_skipped += 1
                continue

            with zipfile.ZipFile(child_path, mode="r") as nested_archive:
                stats.archives_processed += 1
                _bundle_zip_archive(
                    archive=nested_archive,
                    bundle_prefix=bundle_prefix / f"__zip__{child_path.stem}",
                    bundle_zip=bundle_zip,
                    stats=stats,
                    progress=progress,
                    root_alias=root_alias,
                    validate_dicom=validate_dicom,
                    archive_depth=archive_depth + 1,
                    max_archive_depth=max_archive_depth,
                )
            progress.maybe_log(stats, f"{root_alias}: scanning filesystem entries")
            continue

        if _candidate_dicom_suffix(child_path.name):
            if (not validate_dicom) or _is_valid_dicom_path(child_path):
                _write_dicom_path(bundle_zip, child_path, target_arcname, stats, root_alias)
            else:
                stats.skipped_invalid_dicom += 1
            progress.maybe_log(stats, f"{root_alias}: scanning filesystem entries")
            continue

        stats.skipped_non_dicom += 1
        progress.maybe_log(stats, f"{root_alias}: scanning filesystem entries")


def _bundle_root(
    input_root: Path,
    root_alias: str,
    bundle_zip: zipfile.ZipFile,
    stats: BundleStats,
    progress: ProgressState,
    validate_dicom: bool,
    max_archive_depth: int,
) -> None:
    if input_root.is_dir():
        _bundle_directory(
            source_dir=input_root,
            bundle_prefix=Path(root_alias),
            bundle_zip=bundle_zip,
            stats=stats,
            progress=progress,
            root_alias=root_alias,
            validate_dicom=validate_dicom,
            archive_depth=0,
            max_archive_depth=max_archive_depth,
        )
        return

    if input_root.is_file():
        if input_root.name.lower() in METADATA_FILENAMES:
            _write_metadata(bundle_zip, Path(root_alias).joinpath(input_root.name).as_posix(), input_root.read_bytes(), stats, root_alias)
            return

        if zipfile.is_zipfile(input_root):
            stats.archives_seen += 1
            with zipfile.ZipFile(input_root, mode="r") as nested_archive:
                stats.archives_processed += 1
                _bundle_zip_archive(
                    archive=nested_archive,
                    bundle_prefix=Path(root_alias) / f"__zip__{input_root.stem}",
                    bundle_zip=bundle_zip,
                    stats=stats,
                    progress=progress,
                    root_alias=root_alias,
                    validate_dicom=validate_dicom,
                    archive_depth=1,
                    max_archive_depth=max_archive_depth,
                )
            return

        if _candidate_dicom_suffix(input_root.name):
            if (not validate_dicom) or _is_valid_dicom_path(input_root):
                _write_dicom_path(bundle_zip, input_root, Path(root_alias).joinpath(input_root.name).as_posix(), stats, root_alias)
            else:
                stats.skipped_invalid_dicom += 1
            progress.maybe_log(stats, f"{root_alias}: scanning root file")
            return

    stats.skipped_non_dicom += 1


def _build_bundle(
    input_roots: Sequence[Path],
    output_zip_path: Path,
    max_archive_depth: int,
    progress_every: int,
    logger: logging.Logger,
    validate_dicom: bool,
) -> Dict[str, object]:
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    if output_zip_path.exists():
        output_zip_path.unlink()

    stats = BundleStats(input_roots=len(input_roots))
    alias_map: Dict[str, str] = {}
    used_aliases: Set[str] = set()
    progress = ProgressState(logger=logger, progress_every=max(1, int(progress_every)))

    with zipfile.ZipFile(output_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as bundle_zip:
        for root_index, input_root in enumerate(input_roots, start=1):
            root_alias = _make_unique_alias(input_root.name or input_root.stem or "root", used_aliases)
            alias_map[str(input_root)] = root_alias
            logger.info(f"[{root_index}/{len(input_roots)}] Scanning root: {input_root}")
            _bundle_root(
                input_root=input_root,
                root_alias=root_alias,
                bundle_zip=bundle_zip,
                stats=stats,
                progress=progress,
                validate_dicom=validate_dicom,
                max_archive_depth=max_archive_depth,
            )
            logger.info(
                f"[{root_index}/{len(input_roots)}] Finished root: {input_root} | "
                f"dicom_written={stats.dicom_files_written} | metadata_written={stats.metadata_files_written} | "
                f"skipped_non_dicom={stats.skipped_non_dicom} | skipped_invalid_dicom={stats.skipped_invalid_dicom}"
            )

    logger.info(
        f"Bundle build complete: roots={len(input_roots)} | dicom_written={stats.dicom_files_written} | "
        f"metadata_written={stats.metadata_files_written} | archives_seen={stats.archives_seen} | "
        f"archives_processed={stats.archives_processed} | skipped_non_dicom={stats.skipped_non_dicom} | "
        f"skipped_invalid_dicom={stats.skipped_invalid_dicom} | skipped_hidden={stats.skipped_hidden} | "
        f"skipped_traversal={stats.skipped_path_traversal}"
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "output_zip_path": str(output_zip_path),
        "bundle_name": output_zip_path.stem,
        "input_roots": [str(root) for root in input_roots],
        "root_aliases": alias_map,
        "stats": stats.to_summary(),
    }
    return summary


def _upload_to_modal_volume(volume_name: str, local_zip_path: Path, summary_path: Path, remote_base_root: str) -> None:
    try:
        import modal
        import modal._utils.blob_utils as blob_utils
    except Exception as exc:
        raise RuntimeError(
            "Modal is required for upload. Install the Modal requirements or rerun with --skip-upload."
        ) from exc

    blob_utils.HEALTHY_R2_UPLOAD_PERCENTAGE = 1.0

    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    remote_zip_path = f"{remote_base_root}/raw_zips/{local_zip_path.name}"
    remote_summary_path = f"{remote_base_root}/manifests/{summary_path.name}"
    zip_size_mb = local_zip_path.stat().st_size / 1_000_000.0 if local_zip_path.exists() else 0.0
    summary_size_kb = summary_path.stat().st_size / 1_000.0 if summary_path.exists() else 0.0

    upload_logger = logging.getLogger("dicom-bundler")
    upload_logger.info(
        f"Starting Modal upload to volume '{volume_name}' | bundle={local_zip_path.name} ({zip_size_mb:.1f} MB) | "
        f"summary={summary_path.name} ({summary_size_kb:.1f} KB)"
    )
    upload_logger.info(f"Upload target raw zip path: {remote_zip_path}")
    upload_logger.info(f"Upload target summary path: {remote_summary_path}")

    with volume.batch_upload(force=True) as batch:
        upload_logger.info("Uploading raw bundle zip to Modal volume")
        batch.put_file(str(local_zip_path), remote_zip_path)
        upload_logger.info("Raw bundle zip upload staged")
        upload_logger.info("Uploading bundle summary manifest to Modal volume")
        batch.put_file(str(summary_path), remote_summary_path)
        upload_logger.info("Bundle summary upload staged")

    upload_logger.info("Modal volume batch upload completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally upload a single DICOM bundle zip.")
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        required=False,
        help="Input dataset root. Repeat for each downloaded folder or extracted root. Required unless --upload-only is used.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the bundle zip and summary JSON will be written.",
    )
    parser.add_argument(
        "--bundle-name",
        type=str,
        default=DEFAULT_BUNDLE_NAME,
        help="Filename stem for the generated bundle zip.",
    )
    parser.add_argument(
        "--max-archive-depth",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_DEPTH,
        help="Maximum nested ZIP depth to recurse through before skipping deeper archives.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Log scan progress every N processed filesystem/archive entries.",
    )
    parser.add_argument(
        "--validate-dicom",
        action="store_true",
        help="Validate each DICOM candidate with pydicom before bundling. Slower, but stricter.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Skip bundle creation and upload the existing zip and summary from the output root.",
    )
    parser.add_argument(
        "--volume-name",
        type=str,
        default=DEFAULT_VOLUME_NAME,
        help="Modal volume name used when uploading the bundle.",
    )
    parser.add_argument(
        "--remote-base-root",
        type=str,
        default=DEFAULT_REMOTE_BASE_ROOT,
        help="Volume-relative root used for the raw zip upload path.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace any existing local output root.")
    parser.add_argument("--skip-upload", action="store_true", help="Build the bundle locally only.")
    return parser.parse_args()


def _validate_input_roots(input_roots: Sequence[Path]) -> List[Path]:
    normalized_roots: List[Path] = []
    for root in input_roots:
        resolved = root.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Input root not found: {resolved}")
        normalized_roots.append(resolved)
    return normalized_roots


def main() -> None:
    args = parse_args()
    _configure_pydicom_warning_filters()
    logger = _setup_logger()
    output_root = args.output_root.expanduser().resolve()

    bundle_zip_path = output_root / f"{args.bundle_name}.zip"
    summary_path = output_root / "bundle_summary.json"

    if args.upload_only:
        if not bundle_zip_path.exists():
            raise FileNotFoundError(f"Existing bundle zip not found: {bundle_zip_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Existing summary not found: {summary_path}")
        with summary_path.open("r", encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        logger.info(f"Upload-only mode: reusing existing bundle zip at {bundle_zip_path}")
        logger.info(f"Upload-only mode: reusing existing summary at {summary_path}")
    else:
        if not args.input_root:
            raise ValueError("At least one --input-root is required unless --upload-only is used")

        input_roots = _validate_input_roots(args.input_root)

        logger.info(f"Starting DICOM bundle build for {len(input_roots)} root(s)")
        logger.info(f"Output folder: {output_root}")
        logger.info(f"DICOM validation: {'enabled' if args.validate_dicom else 'disabled (fast mode)'}")

        if output_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite to replace it.")
            if output_root.is_file():
                output_root.unlink()
            else:
                import shutil

                shutil.rmtree(output_root)

        output_root.mkdir(parents=True, exist_ok=True)

        summary = _build_bundle(
            input_roots=input_roots,
            output_zip_path=bundle_zip_path,
            max_archive_depth=int(args.max_archive_depth),
            progress_every=int(args.progress_every),
            logger=logger,
            validate_dicom=bool(args.validate_dicom),
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.info(f"Bundle zip written: {bundle_zip_path}")
        logger.info(f"Summary written: {summary_path}")
        logger.info(
            f"Local bundle ready for upload: {bundle_zip_path} ({bundle_zip_path.stat().st_size / 1_000_000.0:.1f} MB)"
        )
        print(json.dumps(summary, indent=2))
        print(json.dumps({"summary_path": str(summary_path)}, indent=2))

    if args.skip_upload:
        logger.info("Skipping Modal upload by request")
        return

    logger.info("Starting Modal bundle upload phase")
    _upload_to_modal_volume(
        volume_name=args.volume_name,
        local_zip_path=bundle_zip_path,
        summary_path=summary_path,
        remote_base_root=args.remote_base_root,
    )

    logger.info("Bundle workflow completed successfully")

    print(json.dumps({"uploaded": True, "volume_name": args.volume_name}, indent=2))


if __name__ == "__main__":
    main()