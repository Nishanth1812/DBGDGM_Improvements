"""
Evaluation Metrics for MM-DBGDGM
Provides specialized metric functions for classification, regression, and survival components.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score
)

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Computes classification performance.
    Args:
        y_true: [N] ground truth labels (0,1,2,3)
        y_pred: [N] predicted labels
        y_prob: [N, num_classes] probabilities
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    if y_prob is not None:
        try:
            # Multi-class AUC (One-vs-Rest)
            metrics['auc_roc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            pass # Fails if not all classes are present in y_true
            
    return metrics

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes regression performance.
    Args:
        y_true: [N] true continuous target
        y_pred: [N] predicted target
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    return metrics

def survival_metrics(event_times_true: np.ndarray, events_true: np.ndarray, expected_times_pred: np.ndarray) -> dict:
    """
    Computes Concordance Index for survival analysis.
    Uses lifelines library.
    
    Args:
        event_times_true: [N] true observed times
        events_true: [N] boolean/int indicating whether event happened (1) or censored (0)
        expected_times_pred: [N] model predicted expected times
    """
    try:
        from lifelines.utils import concordance_index
        c_index = concordance_index(event_times_true, expected_times_pred, events_true)
        return {'c_index': c_index}
    except ImportError:
        return {'c_index': 'lifelines module not installed. Install with: pip install lifelines'}
