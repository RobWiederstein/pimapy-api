# In src/pimapy/nodes/common.py

from typing import List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_abline,
    labs, theme_bw, coord_fixed, theme, element_text
)
import itertools
from sklearn.model_selection import GridSearchCV

# This function should replace the existing `select_champion_model`
# in your project (likely in `src/pimapy/nodes/common.py`)

import logging

log = logging.getLogger(__name__)

def select_champion_model(
    lr_candidate: Dict[str, Any],
    rf_candidate: Dict[str, Any],
    xgb_candidate: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Selects the best model from a set of candidate dictionaries, each containing
    the results from a model tuning run.

    Args:
        lr_candidate: Dictionary with results for Logistic Regression.
        rf_candidate: Dictionary with results for Random Forest.
        xgb_candidate: Dictionary with results for XGBoost.
    """
    candidates = {
        "Logistic Regression": lr_candidate,
        "Random Forest": rf_candidate,
        "XGBoost": xgb_candidate,
    }

    # Find the name of the model with the highest 'best_cv_score'.
    champion_name = max(candidates, key=lambda k: candidates[k].get('best_cv_score', 0))
    
    # Get the dictionary of the winning model.
    champion_result_dict = candidates[champion_name]

    # --- THIS IS THE FIX ---
    # First, get the GridSearchCV object from the dictionary.
    grid_search_object = champion_result_dict.get('model_object')
    
    # Then, extract the actual trained Pipeline from the .best_estimator_ attribute.
    # This is the object your API needs.
    champion_model_pipeline = grid_search_object.best_estimator_
    
    # Also get the best parameters directly from the GridSearchCV object.
    best_hyperparameters = grid_search_object.best_params_
    best_cv_score = champion_result_dict.get('best_cv_score', 0.0)
    # -----------------------

    # Create the info dictionary for logging/saving.
    champion_info = {
        "champion_name": champion_name,
        "best_cv_score": best_cv_score,
        "best_params": best_hyperparameters,
    }

    log.info(f"Champion model selected: {champion_name} with CV score {best_cv_score:.4f}")

    # Return the best scikit-learn Pipeline object and its info.
    return champion_model_pipeline, champion_info

def evaluate_champion_on_test_set(
    champion_model: Pipeline,
    champion_info: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, str]:
    """
    Evaluates the single champion model on the held-out test set.
    """
    now = datetime.now()
    run_id = now.strftime("%Y%m%d%H%M")
    y_pred = champion_model.predict(X_test)
    y_proba = champion_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics_dict = {
        "run_id": run_id,
        "champion_model_name": champion_info["champion_name"],
        "champion_cv_score": champion_info["best_cv_score"],
        "champion_best_params": str(champion_info["best_params"]),
        "final_test_accuracy": accuracy,
    }

    metrics_df = pd.DataFrame([metrics_dict])
    
    return metrics_df, y_test, pd.Series(y_pred, index=y_test.index), run_id


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: pd.Series,
    run_id: str
) -> plt.Figure:
    """
    Given true labels and predicted labels, draw a confusion-matrix heatmap
    using imshow + text. Returns a Matplotlib Figure (dpi=300), including run_id in the title.
    """
    # 1) Turn off interactive mode and close any old figures
    plt.ioff()
    plt.close("all")

    # 2) Compute the 2×2 confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # 3) Create a high-DPI figure
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # 4) Display with imshow to get full control (origin='upper' puts [0,0] in top-left)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", origin="upper")

    # 5) Annotate each cell with its integer value
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j, i,               # column=j, row=i
            f"{cm[i, j]:d}",    # the number
            ha="center", 
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

    # 6) Configure ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([0, 1])
    ax.set_yticklabels([0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (run ID {run_id})")

    # 7) Ensure all cells are inside the axes
    ax.set_xlim(-0.5, cm.shape[1] - 0.5)
    ax.set_ylim(cm.shape[0] - 0.5, -0.5)

    # 8) Tight layout (with a bit more bottom padding for labels)
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()

    # 9) Restore interactive mode
    plt.ion()
    return fig


def plot_multi_model_roc_from_candidates(
    lr_candidate: Dict[str, Any],
    rf_candidate: Dict[str, Any],
    xgb_candidate: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> plt.Figure:
    """
    Takes multiple model candidate dictionaries, computes their ROC curves on the
    test set, and returns a single, combined Plotnine plot.
    """
    # Manually group the candidate dictionaries into a list for processing.
    model_candidates = [lr_candidate, rf_candidate, xgb_candidate]
    roc_dataframes = []

    for candidate in model_candidates:
        model_name = candidate["model_name"]
        model_object = candidate["model_object"]

        y_proba = model_object.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        df = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "model": f"{model_name} (AUC={roc_auc:.3f})"
        })
        roc_dataframes.append(df)

    df_all = pd.concat(roc_dataframes, ignore_index=True)

    p = (
        ggplot(df_all, aes(x="fpr", y="tpr", color="model"))
        + geom_line(size=1.1)
        + geom_abline(linetype="dashed", slope=1, intercept=0, color="gray")
        + coord_fixed(ratio=1, xlim=(0, 1), ylim=(0, 1))
        + theme_bw()
        + labs(
            title="ROC Curves for All Tuned Models on Test Set",
            x="False Positive Rate",
            y="True Positive Rate",
            color="Model"
        )
    )
    fig = p.draw()
    return fig
