import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _subject_feature_vector(subject_embeddings):
    beta_embeddings = np.asarray(subject_embeddings['beta_embeddings']['train'], dtype=np.float32)

    if beta_embeddings.size == 0:
        fallback = []
        for status in ['train', 'valid', 'test']:
            fallback.extend(subject_embeddings['beta_embeddings'][status])
        beta_embeddings = np.asarray(fallback, dtype=np.float32)

    if beta_embeddings.size == 0:
        raise ValueError('No beta embeddings are available to build diagnosis features.')

    return beta_embeddings.mean(axis=(0, 1))


def evaluate_diagnosis_from_embeddings(embeddings, dataset, seed=42):
    features = []
    labels = []

    for subject_idx, _, label in dataset:
        if subject_idx not in embeddings:
            raise KeyError(f'Missing embeddings for subject index {subject_idx}.')
        features.append(_subject_feature_vector(embeddings[subject_idx]))
        labels.append(int(label))

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    present_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError('At least two subjects per class are required for stratified diagnosis evaluation.')

    n_splits = min(5, min_class_count)
    n_repeats = max(1, 10 // n_splits)
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    accuracy_scores = []
    balanced_accuracy_scores = []
    macro_f1_scores = []
    macro_auc_scores = []

    classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', multi_class='auto')),
    ])

    for train_idx, test_idx in splitter.split(X, y):
        classifier.fit(X[train_idx], y[train_idx])
        y_pred = classifier.predict(X[test_idx])

        accuracy_scores.append(float(accuracy_score(y[test_idx], y_pred)))
        balanced_accuracy_scores.append(float(balanced_accuracy_score(y[test_idx], y_pred)))
        macro_f1_scores.append(float(f1_score(y[test_idx], y_pred, average='macro', zero_division=0)))

        fold_classes = np.unique(y[test_idx])
        if len(fold_classes) == len(present_classes):
            y_prob = classifier.predict_proba(X[test_idx])
            macro_auc_scores.append(float(
                roc_auc_score(y[test_idx], y_prob, multi_class='ovr', average='macro')
            ))

    class_count_map = {str(int(class_id)): int(count) for class_id, count in zip(present_classes, class_counts)}

    metrics = {
        'n_subjects': int(len(y)),
        'class_counts': class_count_map,
        'cv_scheme': {
            'n_splits': int(n_splits),
            'n_repeats': int(n_repeats),
            'seed': int(seed),
        },
        'accuracy_mean': float(np.mean(accuracy_scores)),
        'accuracy_std': float(np.std(accuracy_scores)),
        'balanced_accuracy_mean': float(np.mean(balanced_accuracy_scores)),
        'balanced_accuracy_std': float(np.std(balanced_accuracy_scores)),
        'macro_f1_mean': float(np.mean(macro_f1_scores)),
        'macro_f1_std': float(np.std(macro_f1_scores)),
        'macro_auc_ovr_mean': float(np.mean(macro_auc_scores)) if macro_auc_scores else float('nan'),
        'macro_auc_ovr_std': float(np.std(macro_auc_scores)) if macro_auc_scores else float('nan'),
    }

    if min_class_count <= 2:
        metrics['warning'] = (
            'Diagnosis metrics are computed with very few subjects in the rarest class; '
            'treat the mean/std as a low-data estimate, not a stable benchmark.'
        )

    return metrics