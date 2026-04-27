import torch
import numpy as np
from pathlib import Path
from preprocessing.fmri_pipeline import build_fmri_graphs
from preprocessing.smri_pipeline import build_structural_graph
from models.mm_dbgdgm import MM_DBGDGM


def predict_subject(fmri_path, smri_path, model_checkpoint, threshold_sigma=0.5, device=None):
    """
    Load subject data, run forward pass, return prediction dict.

    Args:
        fmri_path: path to .npz file with 'fmri' key (N_roi, T)
        smri_path: path to .npz file with 'smri' key (N_roi, F)
        model_checkpoint: path to .pt model weights
        threshold_sigma: uncertainty threshold for flagging

    Returns:
        dict with predicted_class, probabilities, uncertainty, high_uncertainty_flag, top_attention_regions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    fmri_data = np.load(fmri_path, allow_pickle=True)
    smri_data = np.load(smri_path, allow_pickle=True)

    fmri_ts = fmri_data["fmri"].astype(np.float32)
    smri_feats = smri_data["smri"].astype(np.float32)

    # Z-score
    fmri_ts = (fmri_ts - fmri_ts.mean(axis=1, keepdims=True)) / (fmri_ts.std(axis=1, keepdims=True) + 1e-8)

    fmri_graphs = [g.to(device) for g in build_fmri_graphs(fmri_ts)]
    smri_graph = build_structural_graph(smri_feats).to(device)

    # Load model
    model = MM_DBGDGM().to(device)
    state = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        out = model(fmri_graphs, smri_graph, return_sample=False)

    logits = out["logits"].cpu()
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    uncertainty = out["uncertainty"].item()

    class_names = ["CN", "eMCI", "lMCI", "AD"]
    pred_idx = int(logits.argmax())
    pred_class = class_names[pred_idx]

    # Attention regions (from cross-modal fusion)
    attn_fs = out["attention_fs"]  # (1, num_heads) or averaged
    top_attention_regions = [f"Region_{i}" for i in range(5)]  # placeholder

    result = {
        "predicted_class": pred_class,
        "probabilities": {name: float(probs[i]) for i, name in enumerate(class_names)},
        "uncertainty": float(uncertainty),
        "high_uncertainty_flag": bool(uncertainty > threshold_sigma),
        "top_attention_regions": top_attention_regions,
    }

    print(f"\n  Subject: {Path(fmri_path).stem}")
    print(f"  Predicted: {pred_class} ({probs[pred_idx]:.2%})")
    print(f"  Probabilities: {result['probabilities']}")
    print(f"  Uncertainty (??): {uncertainty:.4f}")

    if result["high_uncertainty_flag"]:
        print("  [WARNING] HIGH UNCERTAINTY -- recommend clinical review")

    return result