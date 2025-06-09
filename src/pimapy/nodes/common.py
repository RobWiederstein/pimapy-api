import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Any
import pandas as pd

def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series,
    run_id: str
) -> plt.Figure:
    """
    Given true labels and predicted labels, draw a confusion-matrix heatmap
    using imshow + text. Returns a Matplotlib Figure (dpi=300), including run_id.
    """
    plt.ioff()
    plt.close("all")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", origin="upper")
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j, i, f"{cm[i, j]:d}",
            fontsize=14, ha="center", va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["TRUE", "FALSE"], fontsize=14)
    ax.set_yticklabels(["TRUE", "FALSE"], fontsize=14)
    ax.set_xlabel("Predicted", fontsize=16)
    ax.set_ylabel("Actual",    fontsize=16)
    ax.set_title( f"Confusion Matrix (run ID {run_id})", fontsize=18)

    ax.set_xlim(-0.5, cm.shape[1] - 0.5)
    ax.set_ylim(cm.shape[0] - 0.5, -0.5)

    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()

    plt.ion()
    return fig

import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    balanced_accuracy_score
)
from typing import Any, Dict

def compute_metrics(
    y_test: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
    tuning_params: Dict[str, Any],
    best_params: Dict[str, Any],
    run_id: str,
    model_name: str
) -> pd.DataFrame:
    """
    Compute a suite of binary-classification metrics and return
    a one-row DataFrame containing:
      - run_id, Model=model_name
      - accuracy, roc_auc, precision, recall, f1_score, matthews_cc, bal_accuracy
      - best_params (stringified)
      - cv_folds, scoring, random_state, stratify
    """
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # Extract tuning metadata
    cv_folds = tuning_params.get("cv")
    scoring = tuning_params.get("scoring")
    random_state = tuning_params.get("random_state")
    stratify_flag = tuning_params.get("stratify", True)

    # Build record
    record = {
        "run_id": run_id,
        "Model": model_name,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "matthews_cc": mcc,
        "bal_accuracy": bal_acc,
        "best_params": str(best_params),
        "cv_folds": cv_folds,
        "scoring": scoring,
        "random_state": random_state,
        "stratify": stratify_flag
    }
    return pd.DataFrame([record])

# Convenience wrappers for specific models

def compute_lr_metrics(*args, **kwargs) -> pd.DataFrame:
    return compute_metrics(*args, model_name="Logistic Regression", **kwargs)


def compute_rf_metrics(*args, **kwargs) -> pd.DataFrame:
    return compute_metrics(*args, model_name="Random Forest", **kwargs)


def compute_xgb_metrics(*args, **kwargs) -> pd.DataFrame:
    return compute_metrics(*args, model_name="XGBoost", **kwargs)
