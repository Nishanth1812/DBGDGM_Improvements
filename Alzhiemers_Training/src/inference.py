from pathlib import Path

import numpy as np
import torch

from .diagnosis import evaluate_diagnosis_from_embeddings
from .utils import prepare_dataset_tensors


def inference(model,
              dataset,
              load_path=Path.cwd() / "models",
              save_path=Path.cwd() / "models",
              valid_prop=0.1, test_prop=0.1,
              device=torch.device("cpu")
              ):
    """
    Performs inference on a dataset using a pre-trained model and saves the results.

    Args:
        model (nn.Module): The pre-trained PyTorch model for inference.
        dataset (list): The dataset on which to perform inference.
        load_path (Path, optional): The directory from where the pre-trained model will be loaded. Default is the current working directory under the 'models' subdirectory.
        save_path (Path, optional): The directory where the results of the inference will be saved. Default is the current working directory under the 'models' subdirectory.
        valid_prop (float, optional): The proportion of the dataset to be used for validation during inference. Default is 0.1.
        test_prop (float, optional): The proportion of the dataset to be used for testing during inference. Default is 0.1.
        device (torch.device, optional): The device to use for model execution (e.g., 'cpu' or 'cuda'). Default is 'cpu'.

    Returns: None. The function saves the results of the inference to the specified save_path. The embeddings, negative log-likelihoods, AUC-ROC, AP are saved. The results are saved as 'results_inference.npy'.
    """

    model_path = load_path if load_path.is_file() else load_path / "checkpoint_best_valid.pt"

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
    except FileNotFoundError:
        print(f'No model found at path: {model_path}')
        return

    model.to(device)
    model.eval()

    if dataset and isinstance(dataset[0][1][0], dict):
        prepare_dataset_tensors(dataset, pin_memory=device.type == 'cuda')

    embeddings = model.predict_embeddings(dataset, valid_prop=valid_prop, test_prop=test_prop)

    edge_nll, edge_aucroc, edge_ap = model.predict_auc_roc_precision(
        dataset,
        valid_prop=valid_prop,
        test_prop=test_prop
    )

    try:
        diagnosis_metrics = evaluate_diagnosis_from_embeddings(embeddings, dataset)
    except Exception as exc:
        diagnosis_metrics = {'error': str(exc)}

    report = (
        f"edge train nll {edge_nll['train']} aucroc {edge_aucroc['train']} ap {edge_ap['train']} | "
        f"edge valid nll {edge_nll['valid']} aucroc {edge_aucroc['valid']} ap {edge_ap['valid']} | "
        f"edge test nll {edge_nll['test']} aucroc {edge_aucroc['test']} ap {edge_ap['test']}"
    )

    print(report)
    if 'error' in diagnosis_metrics:
        print(f"diagnosis evaluation unavailable: {diagnosis_metrics['error']}")
    else:
        print(
            f"diagnosis accuracy {diagnosis_metrics['accuracy_mean']:.4f} +/- {diagnosis_metrics['accuracy_std']:.4f} | "
            f"bal_acc {diagnosis_metrics['balanced_accuracy_mean']:.4f} +/- {diagnosis_metrics['balanced_accuracy_std']:.4f} | "
            f"macro_f1 {diagnosis_metrics['macro_f1_mean']:.4f} +/- {diagnosis_metrics['macro_f1_std']:.4f}"
        )
    print("Saving embeddings.")

    try:
        save_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Saving subject embeddings to {save_path}."
              f"Existing saved results will be overridden.")

    model_save_path = Path(save_path) / "results_inference.npy"

    np.save(model_save_path,
            {
                'edge_nll': edge_nll,
                'edge_aucroc': edge_aucroc,
                'edge_ap': edge_ap,
                'diagnosis_metrics': diagnosis_metrics,
                'embeddings': embeddings
            })

    print("Performance metrics saved.")
