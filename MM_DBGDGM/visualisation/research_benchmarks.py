import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def plot_group_dmn_benchmarks(manifest_path, data_dir, save_path=None):
    """
    Generates a professional comparison of DMN functional connectivity across diagnostic groups.
    Fixed: Now uses NaN-robust averaging to handle subjects with zero-variance ROIs (common in late-stage AD).
    """
    manifest = pd.read_csv(manifest_path)
    data_dir = Path(data_dir)
    
    group_map = {0: 'CN', 1: 'MCI', 3: 'AD'}
    colors_map = {0: '#2ecc71', 1: '#f39c12', 3: '#e74c3c'}
    
    dmn_indices = [34, 35, 66, 67, 22, 23, 24, 25, 64, 65]
    dmn_labels = ["PCC L", "PCC R", "Precu L", "Precu R", "mPFC L", "mPFC R", "mPFC OrbL", "mPFC OrbR", "Ang L", "Ang R"]

    present_groups = sorted([g for g in group_map.keys() if g in manifest['label'].values])
    
    fig, axes = plt.subplots(1, len(present_groups), figsize=(6 * len(present_groups), 5), facecolor='white')
    if len(present_groups) == 1: axes = [axes]
    
    for i, gidx in enumerate(present_groups):
        gname, color, ax = group_map[gidx], colors_map[gidx], axes[i]
        
        group_subs = manifest[manifest['label'] == gidx]
        group_corrs = []
        
        sample_size = min(len(group_subs), 10)
        for _, sub in group_subs.head(sample_size).iterrows():
            f_path = data_dir / sub['fmri_path']
            if f_path.exists():
                try:
                    fmri = np.load(f_path)
                    if fmri.shape[0] >= max(dmn_indices):
                        dmn_ts = fmri[dmn_indices, :]
                        # Correlation coefficient
                        with np.errstate(divide='ignore', invalid='ignore'):
                            corr = np.corrcoef(dmn_ts)
                        group_corrs.append(corr)
                except Exception:
                    continue
        
        if not group_corrs:
            ax.text(0.5, 0.5, "Data unavailable", ha='center', fontsize=12)
        else:
            # Robust averaging: ignore NaNs from zero-variance ROIs
            avg_corr = np.nanmean(group_corrs, axis=0)
            # Fill remaining NaNs (if any) with 0
            avg_corr = np.nan_to_num(avg_corr)
            
            sns.heatmap(avg_corr, annot=True, fmt=".2f", cmap='RdBu_r', vmin=-0.2, vmax=1.0, 
                        ax=ax, cbar=(i == len(present_groups)-1), 
                        xticklabels=dmn_labels if i == 0 else False, 
                        yticklabels=dmn_labels if i == 0 else False)
        
        ax.set_title(f'Group: {gname} (N={sample_size})', fontsize=16, fontweight='bold', color=color, pad=15)
        if i > 0: ax.set_yticks([])
            
    plt.suptitle('Neuro-Diagnostic Benchmark: Default Mode Network (DMN) Connectivity', 
                 fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return save_path
    else:
        return fig
