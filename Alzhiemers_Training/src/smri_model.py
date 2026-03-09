import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn

from .model import LOSS_KEYS


class SmriModel(nn.Module):
    metric_mode = 'classification'
    supports_edge_metrics = False

    def __init__(self, input_shape, embedding_dim=128, num_classes=4, device=torch.device('cpu')):
        super().__init__()

        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if getattr(module, 'bias', None) is not None:
                    nn.init.zeros_(module.bias)

    def _subject_volume(self, subject_data):
        volume = torch.as_tensor(subject_data['volume'], dtype=torch.float32, device=self.device)
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        return volume.unsqueeze(0) if volume.ndim == 4 else volume

    def _encode_volume(self, subject_data):
        volume = self._subject_volume(subject_data)
        features = self.encoder(volume)
        embedding = self.embedding_head(features)
        return embedding.squeeze(0)

    def _classify_subject(self, subject_data):
        embedding = self._encode_volume(subject_data)
        logits = self.classifier(embedding.unsqueeze(0))
        return embedding, logits.squeeze(0)

    def forward(self, batch_data, valid_prop=0.1, test_prop=0.1, temp=1.0, class_weights=None):
        del valid_prop, test_prop, temp
        loss = {key: torch.zeros((), device=self.device) for key in LOSS_KEYS}

        for _, subject_data, subject_label in batch_data:
            _, logits = self._classify_subject(subject_data)
            target = torch.tensor([subject_label], dtype=torch.long, device=self.device)
            weight = class_weights.to(self.device) if class_weights is not None else None
            classification_loss = F.cross_entropy(logits.unsqueeze(0), target, weight=weight)
            loss['classification'] += classification_loss

        return loss

    def predict_auc_roc_precision(self, subject_graphs, valid_prop=0.1, test_prop=0.1):
        del subject_graphs, valid_prop, test_prop
        nan_metrics = {'train': float('nan'), 'valid': float('nan'), 'test': float('nan')}
        return dict(nan_metrics), dict(nan_metrics), dict(nan_metrics)

    def predict_label_metrics(self, subject_graphs, valid_prop=0.1, test_prop=0.1):
        del valid_prop, test_prop

        labels = []
        predictions = []
        probabilities = []
        total_loss = 0.0

        for _, subject_data, subject_label in subject_graphs:
            _, logits = self._classify_subject(subject_data)
            probs = F.softmax(logits, dim=-1)
            target = torch.tensor([subject_label], dtype=torch.long, device=self.device)

            total_loss += F.cross_entropy(logits.unsqueeze(0), target, reduction='sum').item()
            labels.append(int(subject_label))
            predictions.append(int(torch.argmax(probs).item()))
            probabilities.append(probs.detach().cpu().numpy())

        labels = np.asarray(labels, dtype=np.int64)
        predictions = np.asarray(predictions, dtype=np.int64)
        probabilities = np.asarray(probabilities, dtype=np.float32)

        metrics = {
            'loss': total_loss / max(len(subject_graphs), 1),
            'accuracy': float((predictions == labels).mean()) if len(labels) else float('nan'),
            'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)) if len(labels) else float('nan'),
            'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0)) if len(labels) else float('nan'),
            'macro_auc_ovr': float('nan'),
        }

        if len(np.unique(labels)) == self.num_classes and len(labels):
            metrics['macro_auc_ovr'] = float(
                roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
            )

        return metrics

    def predict_embeddings(self, subject_graphs, valid_prop=0.1, test_prop=0.1):
        del valid_prop, test_prop

        subjects = {}
        for subject_idx, subject_data, _ in subject_graphs:
            embedding = self._encode_volume(subject_data).detach().cpu().numpy().astype(np.float32)
            subjects[subject_idx] = {
                'alpha_embedding': embedding,
                'p_c_given_z': {
                    'train': [],
                    'valid': [],
                    'test': [],
                },
                'beta_embeddings': {
                    'train': [embedding[np.newaxis, :]],
                    'valid': [],
                    'test': [],
                },
                'phi_embeddings': {
                    'train': [embedding[np.newaxis, :]],
                    'valid': [],
                    'test': [],
                },
            }

        return subjects