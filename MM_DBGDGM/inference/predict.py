import torch
import numpy as np
from pathlib import Path
from preprocessing.fmri_pipeline import build_fmri_graphs
from preprocessing.smri_pipeline import build_structural_graph
from models.mm_dbgdgm import MM_DBGDGM


def predict_subject(fmri_path, smri_path, model_checkpoint, threshold_sigma=0.5, class_names=None, device=None):
    """
    Load subject data, run forward pass, return prediction dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    fmri_path = Path(fmri_path)
    smri_path = Path(smri_path)

    if fmri_path.suffix == '.npy':
        fmri_ts = np.load(fmri_path).astype(np.float32)
    else:
        fmri_ts = np.load(fmri_path, allow_pickle=True)["fmri"].astype(np.float32)

    if smri_path.suffix == '.npy':
        smri_feats = np.load(smri_path).astype(np.float32)
    else:
        smri_feats = np.load(smri_path, allow_pickle=True)["smri"].astype(np.float32)

    # Z-score
    fmri_ts = (fmri_ts - fmri_ts.mean(axis=1, keepdims=True)) / (fmri_ts.std(axis=1, keepdims=True) + 1e-8)

    fmri_graphs = [g.to(device) for g in build_fmri_graphs(fmri_ts)]
    smri_graph = build_structural_graph(smri_feats).to(device)

    # Load checkpoint and detect num_classes
    state = torch.load(model_checkpoint, map_location=device)
    
    # Detect num_classes from classifier weights
    if "classifier.net.10.weight" in state: # Based on ClassificationHead architecture
        num_classes = state["classifier.net.10.weight"].size(0)
    elif "classifier.net.8.weight" in state: # If architecture changed
        num_classes = state["classifier.net.8.weight"].size(0)
    else:
        # Fallback search for any weight with suffix .weight in the last layer
        keys = [k for k in state.keys() if "classifier" in k and "weight" in k]
        num_classes = state[keys[-1]].size(0)

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    elif len(class_names) != num_classes:
        print(f"  [Warning] Expected {num_classes} class names, got {len(class_names)}. Truncating/padding.")
        class_names = (class_names + [f"Class_{i}" for i in range(num_classes)])[:num_classes]

    # Initialize model with detected num_classes
    model = MM_DBGDGM(num_classes=num_classes).to(device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        out = model(fmri_graphs, smri_graph, return_sample=False)

    logits = out["logits"].cpu()
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    uncertainty = out["uncertainty"].item()

    pred_idx = int(logits.argmax())
    pred_class = class_names[pred_idx]

    # Attention regions (from cross-modal fusion)
    attn_fs = out["attention_fs"].cpu().numpy()[0]
    attn_sf = out["attention_sf"].cpu().numpy()[0]
    
    # Ensure weights are 1D for sorting (average across layers/heads if necessary)
    if attn_fs.ndim > 1:
        attn_fs_1d = np.mean(attn_fs, axis=0)
    else:
        attn_fs_1d = attn_fs

    # Sort regions by weight
    top_indices = np.argsort(attn_fs_1d)[::-1][:10]
    top_attention_regions = [f"ROI_{i+1}" for i in top_indices]

    result = {
        "predicted_class": pred_class,
        "confidence": float(probs[pred_idx]),
        "probabilities": {name: float(probs[i]) for i, name in enumerate(class_names)},
        "uncertainty": float(uncertainty),
        "high_uncertainty_flag": bool(uncertainty > threshold_sigma),
        "top_attention_regions": top_attention_regions,
        "attention_weights": {
            "fs": attn_fs.tolist(),
            "sf": attn_sf.tolist()
        }
    }

    print(f"\n  Subject: {fmri_path.stem}")
    print(f"  Predicted: {pred_class} ({probs[pred_idx]:.2%})")
    print(f"  Uncertainty (??): {uncertainty:.4f}")

    if result["high_uncertainty_flag"]:
        print("  [WARNING] HIGH UNCERTAINTY -- recommend clinical review")

    return result