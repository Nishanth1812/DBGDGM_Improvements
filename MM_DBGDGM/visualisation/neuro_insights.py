import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_dmn_connectivity(fmri_ts, subject_id, save_path=None):
    """
    Visualizes the functional connectivity within the Default Mode Network (DMN).
    """
    dmn_indices = [34, 35, 66, 67, 22, 23, 24, 25, 64, 65]
    dmn_labels = [
        "PCC (L)", "PCC (R)", "Precuneus (L)", "Precuneus (R)",
        "mPFC (L)", "mPFC (R)", "mPFC (Orb L)", "mPFC (Orb R)",
        "Angular (L)", "Angular (R)"
    ]
    
    # Extract timeseries for DMN nodes
    # If fmri_ts is (90, T)
    if fmri_ts.shape[0] < max(dmn_indices):
        print(f"  [Warning] Timeseries size {fmri_ts.shape[0]} too small for DMN indices.")
        return None
        
    dmn_ts = fmri_ts[dmn_indices, :]
    corr_matrix = np.corrcoef(dmn_ts)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=dmn_labels, yticklabels=dmn_labels, ax=ax,
                cbar_kws={'label': 'Functional Connectivity (Pearson R)'})
    
    ax.set_title(f"DMN Functional Connectivity Map - Subject: {subject_id}", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    
    fig.text(0.1, -0.05, 
             "Clinical Note: Reduced connectivity in PCC/Precuneus nodes (lower correlation values) \n"
             "is a signature biomarker for early-stage Alzheimer's and MCI.", 
             fontsize=10, style='italic', color='#555', bbox=dict(facecolor='#f8f9fa', alpha=0.8, edgecolor='none'))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        return fig

def plot_multimodal_contrast(f_attn, s_attn, subject_id, save_path=None):
    """
    Shows a bar chart comparing Functional vs Structural attention.
    Adapts to both ROI-level and Latent Head-level attention.
    """
    f_weights = np.array(f_attn)
    s_weights = np.array(s_attn)
    
    # Average across heads/layers if 2D
    if f_weights.ndim > 1: f_weights = np.mean(f_weights, axis=0)
    if s_weights.ndim > 1: s_weights = np.mean(s_weights, axis=0)
    
    num_elements = len(f_weights)
    dmn_indices = [34, 35, 66, 67, 22, 23, 24, 25, 64, 65]
    
    # Case 1: ROI-level attention (90+ ROIs)
    if num_elements >= 90:
        labels = ["PCC L", "PCC R", "Precu L", "Precu R", "mPFC L", "mPFC R", "mPFC OrbL", "mPFC OrbR", "Ang L", "Ang R"]
        f_data = f_weights[dmn_indices]
        s_data = s_weights[dmn_indices]
        title_suffix = "(Default Mode Network Analysis)"
    # Case 2: Latent Head attention (e.g. 8 heads)
    else:
        labels = [f"Head {i+1}" for i in range(num_elements)]
        f_data = f_weights
        s_data = s_weights
        title_suffix = "(Latent Cross-Modal Attention)"
        
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, f_data, width, label='Functional Attention', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, s_data, width, label='Structural Attention', color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Model Attention Weight')
    ax.set_title(f'Multi-modal Sensitivity Contrast {title_suffix}\nSubject: {subject_id}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if num_elements > 10 else 0)
    ax.legend()
    
    if num_elements >= 90:
        ax.text(0.02, 0.95, "Interpretation: High fMRI bars on DMN nodes suggest early-stage signatures\n"
                "detected before physical atrophy.", 
                transform=ax.transAxes, fontsize=10, style='italic', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        return fig
