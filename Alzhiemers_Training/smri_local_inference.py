import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.dataset import load_dataset
from src.diagnosis import evaluate_diagnosis_from_embeddings
from src.smri_model import SmriModel


CLASS_NAMES = {
    0: 'Non Demented',
    1: 'Very mild Dementia',
    2: 'Mild Dementia',
    3: 'Moderate Dementia',
}


def summarize_predictions(predictions: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in predictions:
        label_name = item['predicted_label_name']
        counts[label_name] = counts.get(label_name, 0) + 1
    return counts


def print_results(summary: dict, predictions: list[dict], preview_count: int = 10) -> None:
    print('sMRI Inference Results')
    print('=' * 80)
    print(f"Checkpoint: {summary['checkpoint_path']}")
    print(f"Device: {summary['device']}")
    print(f"Subjects: {summary['num_subjects']}")
    print(f"Input shape: {summary['input_shape']}")
    print(f"Embedding dim: {summary['embedding_dim']}")
    print(f"Classes: {summary['num_classes']}")
    print()

    label_metrics = summary['label_metrics']
    print('Classification Metrics')
    print('-' * 80)
    print(f"Loss: {label_metrics['loss']:.4f}")
    print(f"Accuracy: {label_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {label_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {label_metrics['macro_f1']:.4f}")
    print(f"Macro AUC OVR: {label_metrics['macro_auc_ovr']:.4f}")
    print()

    diagnosis_metrics = summary['diagnosis_metrics']
    print('Diagnosis Metrics')
    print('-' * 80)
    if 'error' in diagnosis_metrics:
        print(f"Unavailable: {diagnosis_metrics['error']}")
    else:
        print(f"Accuracy mean/std: {diagnosis_metrics['accuracy_mean']:.4f} / {diagnosis_metrics['accuracy_std']:.4f}")
        print(
            f"Balanced accuracy mean/std: {diagnosis_metrics['balanced_accuracy_mean']:.4f} / "
            f"{diagnosis_metrics['balanced_accuracy_std']:.4f}"
        )
        print(f"Macro F1 mean/std: {diagnosis_metrics['macro_f1_mean']:.4f} / {diagnosis_metrics['macro_f1_std']:.4f}")
        print(f"Macro AUC OVR mean/std: {diagnosis_metrics['macro_auc_ovr_mean']:.4f} / {diagnosis_metrics['macro_auc_ovr_std']:.4f}")
    print()

    predicted_counts = summarize_predictions(predictions)
    print('Predicted Class Counts')
    print('-' * 80)
    for class_name, count in predicted_counts.items():
        print(f"{class_name}: {count}")
    print()

    print(f'Subject Preview (first {min(preview_count, len(predictions))})')
    print('-' * 80)
    for item in predictions[:preview_count]:
        print(
            f"subject={item['subject_idx']:>3} | true={item['true_label_name']:<20} | "
            f"pred={item['predicted_label_name']:<20} | conf={item['confidence']:.4f}"
        )
    print('=' * 80)


def resolve_checkpoint(workspace_root: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (workspace_root / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        return checkpoint_path

    candidates = [
        workspace_root / 'local_results' / 'run 2' / 'checkpoint_best_valid_model_only.pt',
        workspace_root / 'local_results' / 'run 2' / 'checkpoint_latest_model_only.pt',
        workspace_root / 'local_results' / 'models_oasis_1' / 'checkpoint_best_valid.pt',
        workspace_root / 'local_results' / 'trainingresults' / 'models_oasis_1' / 'checkpoint_best_valid.pt',
        workspace_root / 'local_results' / 'trainingresults' / 'models_oasis_1' / 'checkpoint_latest.pt',
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError('No compatible checkpoint was found under local_results/.')


def load_model_state(checkpoint_path: Path, device: torch.device) -> tuple[dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        return checkpoint['model_state'], checkpoint

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict'], checkpoint

    if isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, {'checkpoint_type': 'model_only'}

    raise ValueError(f'Unsupported checkpoint format: {checkpoint_path}')


def infer_model_dimensions(model_state: dict) -> tuple[int, int]:
    classifier_weight = model_state.get('classifier.weight')
    if classifier_weight is None:
        raise KeyError('classifier.weight is missing from the checkpoint state dict.')

    num_classes = int(classifier_weight.shape[0])
    embedding_dim = int(classifier_weight.shape[1])
    return embedding_dim, num_classes


def predict_subject(model: SmriModel, subject_data: dict) -> tuple[np.ndarray, np.ndarray]:
    embedding, logits = model._classify_subject(subject_data)
    probabilities = torch.softmax(logits, dim=-1)
    return (
        embedding.detach().cpu().numpy().astype(np.float32),
        probabilities.detach().cpu().numpy().astype(np.float32),
    )


def build_predictions(model: SmriModel, dataset: list) -> tuple[list[dict], dict, dict]:
    predictions = []
    embeddings = model.predict_embeddings(dataset)
    label_metrics = model.predict_label_metrics(dataset)

    for subject_idx, subject_data, label in dataset:
        embedding, probabilities = predict_subject(model, subject_data)
        predicted_label = int(np.argmax(probabilities))
        predictions.append({
            'subject_idx': int(subject_idx),
            'true_label': int(label),
            'true_label_name': CLASS_NAMES.get(int(label), str(label)),
            'predicted_label': predicted_label,
            'predicted_label_name': CLASS_NAMES.get(predicted_label, str(predicted_label)),
            'confidence': float(probabilities[predicted_label]),
            'probabilities': probabilities.tolist(),
            'embedding': embedding.tolist(),
        })

    try:
        diagnosis_metrics = evaluate_diagnosis_from_embeddings(embeddings, dataset)
    except Exception as exc:
        diagnosis_metrics = {'error': str(exc)}

    return predictions, label_metrics, diagnosis_metrics


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    parser = argparse.ArgumentParser(description='Small local sMRI inference pipeline for the OASIS model.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file path. Defaults to auto-discovery under local_results/.')
    parser.add_argument('--data-dir', type=str, default=str(script_dir / 'data'), help='OASIS data directory used by the existing dataset loader.')
    parser.add_argument('--output-dir', type=str, default=str(workspace_root / 'local_results' / 'smri_inference'), help='Directory for saved inference outputs.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Torch device used for inference.')
    parser.add_argument('--image-size', type=int, default=128, help='Slice resize used by the cached sMRI dataset builder.')
    parser.add_argument('--num-slices', type=int, default=64, help='Number of slices sampled into each sMRI volume.')
    args = parser.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    checkpoint_path = resolve_checkpoint(workspace_root, args.checkpoint)
    model_state, checkpoint_meta = load_model_state(checkpoint_path, device)

    dataset = load_dataset(
        dataset='oasis',
        data_mode='smri',
        data_dir=args.data_dir,
        image_size=args.image_size,
        num_slices=args.num_slices,
    )
    if not dataset:
        raise ValueError('Dataset is empty. Check Alzhiemers_Training/data and the cached OASIS sMRI dataset.')

    input_shape = tuple(int(dim) for dim in dataset[0][1]['volume'].shape)
    embedding_dim, num_classes = infer_model_dimensions(model_state)

    model = SmriModel(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        device=device,
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    predictions, label_metrics, diagnosis_metrics = build_predictions(model, dataset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'checkpoint_path': str(checkpoint_path),
        'device': str(device),
        'num_subjects': int(len(dataset)),
        'input_shape': list(input_shape),
        'embedding_dim': embedding_dim,
        'num_classes': num_classes,
        'label_metrics': label_metrics,
        'diagnosis_metrics': diagnosis_metrics,
        'checkpoint_metadata_keys': sorted(list(checkpoint_meta.keys())) if isinstance(checkpoint_meta, dict) else [],
    }

    summary_path = output_dir / 'summary.json'
    predictions_path = output_dir / 'subject_predictions.json'
    raw_path = output_dir / 'results_inference.npy'

    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    with open(predictions_path, 'w', encoding='utf-8') as handle:
        json.dump(predictions, handle, indent=2)

    np.save(
        raw_path,
        np.array({
            'summary': summary,
            'predictions': predictions,
        }, dtype=object),
        allow_pickle=True,
    )

    print_results(summary, predictions)
    print(f'Saved subject predictions to {predictions_path}')
    print(f'Saved summary to {summary_path}')
    print(f'Saved raw inference bundle to {raw_path}')


if __name__ == '__main__':
    main()