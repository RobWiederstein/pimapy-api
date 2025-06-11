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
# Get a logger for this file
log = logging.getLogger(__name__)


def select_champion_model(
    *model_candidates: Dict[str, Any]
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Selects the best model from a list of candidates based on 'best_cv_score'.
    """
    if not model_candidates:
        raise ValueError("Model candidate list cannot be empty.")

    champion_info_full = max(model_candidates, key=lambda x: x['best_cv_score'])
    
    champion_model = champion_info_full["model_object"]
    champion_info_serializable = champion_info_full.copy()
    del champion_info_serializable["model_object"]

    log.info("--- Model Selection Results ---")
    for cand in model_candidates:
        log.info(f"  - Model: {cand['model_name']}, CV Score: {cand['best_cv_score']:.4f}")

    log.info(f">> Champion Model: {champion_info_serializable['model_name']} <<")

    return champion_model, champion_info_serializable


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
        "champion_model_name": champion_info["model_name"],
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


import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_errorbar, geom_point,
    labs, theme_bw, theme, element_text
)


# def plot_cv_scores_from_candidates(
#     lr_candidate: Dict[str, Any],
#     rf_candidate: Dict[str, Any],
#     xgb_candidate: Dict[str, Any]
# ) -> plt.Figure:
#     """
#     Takes multiple model candidate dicts (each a fitted GridSearchCV or similar),
#     extracts the fold-by-fold test scores for the best parameter setting,
#     and plots mean ± [min, max] as an error bar.
#     """
#     log.info("🛠️  Running updated plot_cv_scores_from_candidates with error bars")
#     candidates = [lr_candidate, rf_candidate, xgb_candidate]
#     stats = []

#     for cand in candidates:
#         name = cand["model_name"]
#         search = cand["model_object"]  # e.g. a GridSearchCV
#         cv_res = search.cv_results_
#         best_i = search.best_index_

#         # find all 'splitN_test_score' keys, sort by N
#         split_keys = sorted(
#             [k for k in cv_res if k.startswith("split") and k.endswith("_test_score")],
#             key=lambda s: int(s.split("split")[1].split("_")[0])
#         )
#         scores = np.array([cv_res[k][best_i] for k in split_keys])

#         stats.append({
#             "Model": name,
#             "mean": scores.mean(),
#             "min":   scores.min(),
#             "max":   scores.max()
#         })

#     df = pd.DataFrame(stats)
#     # order by descending mean
#     order = df.sort_values("mean", ascending=False)["Model"]
#     df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)

#     p = (
#         ggplot(df, aes(x="Model"))
#         + geom_errorbar(aes(ymin="min", ymax="max"), width=0.2)
#         + geom_point(aes(y="mean"), size=4)
#         + labs(
#             title="Model CV Performance (mean ± range)",
#             x="Model",
#             y="ROC-AUC"
#         )
#         + theme_bw()
#         + theme(axis_text_x=element_text(rotation=15, hjust=1))
#     )
#     return p.draw()
