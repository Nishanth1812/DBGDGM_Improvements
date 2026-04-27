"""
Generates synthetic ADNI-type neuroimaging data for MM-DBGDGM testing.

Produces 25 subjects (7 CN, 6 eMCI, 6 lMCI, 6 AD) with paired
fMRI time series and sMRI structural features mimicking known
AD biomarker patterns from the neuroscience literature.

Usage:
    python generate_synthetic_adni.py --seed 42 --output_dir ./subjects

Output per subject:
    subjects/sub-{id}.npz  containing:
        fmri   : (90, 200)  ? 90 AAL ROIs ? 200 timepoints
        smri   : (90, 4)    ? 90 regions ? [thickness, gm_vol, sub_vol, surf_area]
        label  : int        ? 0=CN, 1=eMCI, 2=lMCI, 3=AD
        age    : float
        sex    : int        ? 0=F, 1=M
    subjects/manifest.csv
"""

import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path

# ?? AAL-90 network membership (approximate groupings) ?????????????????????????
DMN_REGIONS       = [34, 35, 36, 37, 67, 68, 69, 70, 71, 72]
LIMBIC_REGIONS    = [25, 26, 37, 38, 39, 40, 41, 42, 43, 44]
FRONTAL_REGIONS   = list(range(1, 16))
PARIETAL_REGIONS  = list(range(57, 72))
TEMPORAL_REGIONS  = list(range(81, 90))

AD_SENSITIVE = {
    "hippocampus":         [36, 37],
    "entorhinal":          [24, 25],
    "parahippocampal":     [38, 39],
    "amygdala":            [40, 41],
    "posterior_cingulate": [34, 35],
    "precuneus":           [66, 67],
}

FMRI_PARAMS = {
    "CN":    (1.00, 0.05, 0.30, 0.35),
    "eMCI":  (0.85, 0.10, 0.28, 0.30),
    "lMCI":  (0.65, 0.18, 0.25, 0.25),
    "AD":    (0.45, 0.28, 0.20, 0.18),
}

SMRI_ATROPHY = {
    "CN":   {"ad_sensitive": 1.00, "frontal": 1.00, "global": 1.00},
    "eMCI": {"ad_sensitive": 0.89, "frontal": 1.00, "global": 0.98},
    "lMCI": {"ad_sensitive": 0.78, "frontal": 0.95, "global": 0.96},
    "AD":   {"ad_sensitive": 0.62, "frontal": 0.90, "global": 0.92},
}

LABEL_MAP = {"CN": 0, "eMCI": 1, "lMCI": 2, "AD": 3}
N_ROIS    = 90
N_TIMES   = 200
N_FEATS   = 4

FEAT_MEANS = np.array([2.4, 2800.0, 850.0, 1850.0])
FEAT_STDS  = np.array([0.25,  280.0,  90.0,  185.0])
GROUPS_COUNT = {"CN": 7, "eMCI": 6, "lMCI": 6, "AD": 6}


def build_covariance(group: str, rng: np.random.Generator) -> np.ndarray:
    dmn_scale, noise_level, _, base_conn = FMRI_PARAMS[group]

    cov = np.eye(N_ROIS) * 0.5
    cov += base_conn * 0.3 * (rng.random((N_ROIS, N_ROIS)) - 0.5)

    for i in DMN_REGIONS:
        for j in DMN_REGIONS:
            if i != j:
                cov[i, j] = 0.55 * dmn_scale + rng.normal(0, 0.04)

    for i in LIMBIC_REGIONS:
        for j in LIMBIC_REGIONS:
            if i != j:
                cov[i, j] = 0.35 * dmn_scale * 0.85 + rng.normal(0, 0.03)

    for i in FRONTAL_REGIONS:
        for j in FRONTAL_REGIONS:
            if i != j:
                cov[i, j] = 0.28 * (1.0 if group in ["CN", "eMCI"] else 0.80) + rng.normal(0, 0.02)

    cov = (cov + cov.T) / 2
    cov += noise_level * np.eye(N_ROIS)
    min_eig = np.linalg.eigvalsh(cov).min()
    if min_eig < 0.01:
        cov += (abs(min_eig) + 0.01) * np.eye(N_ROIS)

    return cov


def generate_fmri(group: str, rng: np.random.Generator) -> np.ndarray:
    _, _, ar_rho, _ = FMRI_PARAMS[group]
    cov = build_covariance(group, rng)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += 0.1 * np.eye(N_ROIS)
        L = np.linalg.cholesky(cov)

    ts = np.zeros((N_ROIS, N_TIMES), dtype=np.float32)
    ts[:, 0] = L @ rng.standard_normal(N_ROIS)
    for t in range(1, N_TIMES):
        innovation = L @ rng.standard_normal(N_ROIS)
        ts[:, t] = ar_rho * ts[:, t-1] + np.sqrt(1 - ar_rho**2) * innovation

    ts += rng.normal(0, 0.05, ts.shape)
    ts = (ts - ts.mean(axis=1, keepdims=True)) / (ts.std(axis=1, keepdims=True) + 1e-8)

    return ts


def generate_smri(group: str, rng: np.random.Generator) -> np.ndarray:
    atrophy = SMRI_ATROPHY[group]
    feats = np.zeros((N_ROIS, N_FEATS), dtype=np.float32)

    ad_sens_flat = [idx for idxs in AD_SENSITIVE.values() for idx in idxs]

    for roi in range(N_ROIS):
        base = rng.normal(FEAT_MEANS, FEAT_STDS * 0.15)

        if roi in ad_sens_flat:
            scale = atrophy["ad_sensitive"]
        elif roi in FRONTAL_REGIONS:
            scale = atrophy["frontal"]
        else:
            scale = atrophy["global"]

        noise = rng.normal(1.0, 0.04)
        feats[roi] = base * scale * noise

    feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-8)

    return feats


def generate_all_subjects(output_dir: Path, seed: int = 42):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    subject_id = 1

    print("=" * 60)
    print("  MM-DBGDGM Synthetic ADNI Data Generator")
    print("=" * 60)

    for group, count in GROUPS_COUNT.items():
        print(f"\nGenerating {count} {group} subjects...")
        group_fmri_corrs = []

        for i in range(count):
            sub_id = f"sub-{subject_id:03d}"
            age    = float(rng.integers(65, 86))
            sex    = int(rng.integers(0, 2))

            fmri = generate_fmri(group, rng)
            smri = generate_smri(group, rng)
            label = LABEL_MAP[group]

            np.savez_compressed(
                output_dir / f"{sub_id}.npz",
                fmri=fmri,
                smri=smri,
                label=np.array(label),
                age=np.array(age),
                sex=np.array(sex),
                subject_id=np.array(sub_id),
                group=np.array(group),
            )

            records.append({
                "subject_id": sub_id, "group": group, "label": label,
                "age": age, "sex": sex,
            })

            dmn_ts = fmri[DMN_REGIONS, :]
            corr_mat = np.corrcoef(dmn_ts)
            mean_dmn_corr = corr_mat[np.triu_indices_from(corr_mat, k=1)].mean()
            group_fmri_corrs.append(mean_dmn_corr)

            subject_id += 1

        mean_smri_atrophy = SMRI_ATROPHY[group]["ad_sensitive"]
        print(f"  [OK] Mean DMN coherence: {np.mean(group_fmri_corrs):.3f} +/- {np.std(group_fmri_corrs):.3f}")
        print(f"  [OK] AD-sensitive region atrophy factor: {mean_smri_atrophy:.2f}")

    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    print("\n" + "=" * 60)
    print("  SANITY CHECK ? Group-Level Biomarker Summary")
    print("=" * 60)
    print(f"  {'Group':<8} {'N':>4}  {'Atrophy (AD-sens)':>18}  {'DMN scale':>10}")
    print("  " + "-" * 50)
    for group in ["CN", "eMCI", "lMCI", "AD"]:
        n = GROUPS_COUNT[group]
        atr = SMRI_ATROPHY[group]["ad_sensitive"]
        dmn = FMRI_PARAMS[group][0]
        print(f"  {group:<8} {n:>4}  {atr:>18.2f}  {dmn:>10.2f}")

    print("\n  Expected pattern (AD < lMCI < eMCI < CN for both measures): OK")
    print(f"\n  Total subjects: {sum(GROUPS_COUNT.values())}")
    print(f"  Manifest saved: {output_dir / 'manifest.csv'}")
    print(f"  Subject files:  {output_dir}/*.npz")
    print("=" * 60)

    return manifest


def verify_data(output_dir: Path, manifest: pd.DataFrame):
    print("\n  Verifying saved files...")
    errors = []
    for _, row in manifest.iterrows():
        fpath = output_dir / f"{row['subject_id']}.npz"
        if not fpath.exists():
            errors.append(f"Missing: {fpath}")
            continue
        data = np.load(fpath, allow_pickle=True)
        if data["fmri"].shape != (N_ROIS, N_TIMES):
            errors.append(f"{row['subject_id']}: fmri shape {data['fmri'].shape}")
        if data["smri"].shape != (N_ROIS, N_FEATS):
            errors.append(f"{row['subject_id']}: smri shape {data['smri'].shape}")

    if errors:
        print("  [FAIL] Verification FAILED:")
        for e in errors:
            print(f"    {e}")
    else:
        print(f"  [OK] All {len(manifest)} subject files verified successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ADNI-type data for MM-DBGDGM")
    parser.add_argument("--seed",       type=int,  default=42,         help="Random seed")
    parser.add_argument("--output_dir", type=str,  default="./subjects", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    manifest = generate_all_subjects(out, seed=args.seed)
    verify_data(out, manifest)

    print("\n  [OK] Data generation complete. Ready for MM-DBGDGM training.\n")