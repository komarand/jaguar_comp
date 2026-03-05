import numpy as np
from sklearn.metrics import average_precision_score

def compute_identity_balanced_map(y_true, y_scores, identities):
    """
    Compute Identity-balanced Mean Average Precision (mAP).
    Calculates AP for each unique identity separately and takes the mean.

    Args:
        y_true (list or np.array): Ground truth binary labels (1 for match, 0 for non-match)
        y_scores (list or np.array): Predicted scores/probabilities (higher is better match)
        identities (list or np.array): Identity labels for each pair (typically the query identity)

    Returns:
        float: Identity-balanced mAP
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    identities = np.array(identities)

    unique_ids = np.unique(identities)
    aps = []

    for uid in unique_ids:
        mask = (identities == uid)
        true_labels = y_true[mask]
        scores = y_scores[mask]

        # Check if there is at least one positive example for the identity
        if len(np.unique(true_labels)) > 1:
            ap = average_precision_score(true_labels, scores)
            aps.append(ap)

    if not aps:
        print("Warning: No valid AP calculations (needs both positive and negative samples).")
        return 0.0

    return np.mean(aps)

def compute_map(y_true, y_scores):
    """Standard mean average precision"""
    return average_precision_score(y_true, y_scores)
