# utils/metrics.py

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_accuracy(labels, predictions):
    """
    Calculates the accuracy score.

    Args:
        labels (List[int]): True labels.
        predictions (List[int]): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    return accuracy_score(labels, predictions) * 100.0

def calculate_f1_score(labels, predictions, average='weighted'):
    """
    Calculates the F1 score.

    Args:
        labels (List[int]): True labels.
        predictions (List[int]): Predicted labels.
        average (str): Type of averaging performed on the data.

    Returns:
        float: F1 score.
    """
    return f1_score(labels, predictions, average=average, zero_division=0)

def calculate_precision(labels, predictions, average='weighted'):
    """
    Calculates the precision score.

    Args:
        labels (List[int]): True labels.
        predictions (List[int]): Predicted labels.
        average (str): Type of averaging performed on the data.

    Returns:
        float: Precision score.
    """
    return precision_score(labels, predictions, average=average, zero_division=0)

def calculate_recall(labels, predictions, average='weighted'):
    """
    Calculates the recall score.

    Args:
        labels (List[int]): True labels.
        predictions (List[int]): Predicted labels.
        average (str): Type of averaging performed on the data.

    Returns:
        float: Recall score.
    """
    return recall_score(labels, predictions, average=average, zero_division=0)
