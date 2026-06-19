import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


def per_class_accuracy(y_true, y_pred, num_classes=7):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = {}

    for class_id in range(num_classes):
        mask = y_true == class_id

        if mask.sum() == 0:
            result[class_id] = None
        else:
            result[class_id] = (y_pred[mask] == y_true[mask]).mean()

    return result