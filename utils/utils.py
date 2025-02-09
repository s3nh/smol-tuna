import numpy as np 
import torch


def compute_metrics(p: EvalPrediction):
    """
    Function to compute evaluation metrics.
    """
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids

    # Flatten the predictions and labels for accurate computation of metrics
    preds = preds.flatten()
    labels = labels.flatten()

    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}
