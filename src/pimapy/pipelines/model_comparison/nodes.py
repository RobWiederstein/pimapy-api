# In src/pimapy/pipelines/model_comparison/nodes.py

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from plotnine import (
    ggplot, aes, geom_point, geom_line, geom_errorbar, geom_abline,
    labs, scale_color_manual, theme_bw, coord_fixed, theme, element_text
)
import logging
log = logging.getLogger(__name__)

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
    plt.close(fig)
    return fig

def plot_cv_scores_from_candidates(
    lr_candidate: Dict[str, Any],
    rf_candidate: Dict[str, Any],
    xgb_candidate: Dict[str, Any]
) -> plt.Figure:
    """
    Takes multiple model candidate dicts (each a fitted GridSearchCV or similar),
    extracts the fold-by-fold test scores for the best parameter setting,
    and plots mean ± [min, max] as an error bar.
    """
    log.info("🛠️  Running updated plot_cv_scores_from_candidates with error bars")

    candidates = [lr_candidate, rf_candidate, xgb_candidate]
    stats = []

    for cand in candidates:
        name = cand["model_name"]
        search = cand["model_object"]  # e.g. a GridSearchCV
        cv_res = search.cv_results_
        best_i = search.best_index_

        # find all 'splitN_test_score' keys, sort by N
        split_keys = sorted(
            [k for k in cv_res if k.startswith("split") and k.endswith("_test_score")],
            key=lambda s: int(s.split("split")[1].split("_")[0])
        )
        scores = np.array([cv_res[k][best_i] for k in split_keys])

        stats.append({
            "Model": name,
            "mean": scores.mean(),
            "min":   scores.min(),
            "max":   scores.max()
        })

    df = pd.DataFrame(stats)
    # order by descending mean
    order = df.sort_values("mean", ascending=False)["Model"]
    df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)

    p = (
        ggplot(df, aes(x="Model"))
        + geom_errorbar(aes(ymin="min", ymax="max"), width=0.2)
        + geom_point(aes(y="mean"), size=4)
        + labs(
            title="Model CV Performance (mean ± range)",
            x="Model",
            y="ROC-AUC"
        )
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=15, hjust=1))
    )
    return p.draw()

# def plot_cv_scores_from_candidates(
#     lr_candidate: Dict[str, Any],
#     rf_candidate: Dict[str, Any],
#     xgb_candidate: Dict[str, Any]
# ) -> plt.Figure:
#     """
#     Takes multiple model candidate dictionaries and creates a point plot
#     comparing their `best_cv_score`.
#     """
#     model_candidates = [lr_candidate, rf_candidate, xgb_candidate]

#     comparison_data = [
#         {"Model": cand["model_name"], "CV Score": cand["best_cv_score"]}
#         for cand in model_candidates
#     ]
#     df_compare = pd.DataFrame(comparison_data)

#     # Sort by score to determine the desired plot order
#     df_compare = df_compare.sort_values("CV Score", ascending=False)

#     # --- THIS IS THE FIX ---
#     # Manually set the order of the 'Model' column by making it a categorical type.
#     # This replaces the need for the reorder() function in the ggplot call.
#     model_order = df_compare["Model"].tolist()
#     df_compare["Model"] = pd.Categorical(df_compare["Model"], categories=model_order, ordered=True)

#     # Now, use the correctly ordered 'Model' column directly in the aesthetic mapping.
#     p = (
#         ggplot(df_compare, aes(x="Model", y="CV Score"))
#         + geom_point(size=4, color="#0072B2")
#         + labs(
#             title="Model Performance Comparison (Cross-Validation)",
#             x="Model",
#             y="Mean CV Score (roc_auc)"
#         )
#         + theme_bw()
#         + theme(axis_text_x=element_text(rotation=15, hjust=1))
#     )
#     fig = p.draw()
#     plt.close(fig)
#     return fig
