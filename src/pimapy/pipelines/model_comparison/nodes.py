# src/<your_package>/pipelines/model_comparison/nodes.py

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from plotnine import (
    ggplot, aes, coord_fixed, geom_point, geom_errorbar, geom_line, geom_abline,
    facet_wrap, labs, scale_color_manual, theme_bw, theme, element_text
)
from typing import Any

def plot_multi_models_roc(
    tuned_lr_model: Pipeline,
    tuned_rf_model: Pipeline,
    tuned_xgb_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Any:
    """
    Take two fitted pipelines (LogisticRegression and RandomForestClassifier),
    compute ROC curves on the same test set, and return a Plotnine plot
    with both ROC curves overlaid at a 1:1 aspect ratio.
    """
    # 1) Get predicted probabilities for the positive class
    y_proba_lr = tuned_lr_model.predict_proba(X_test)[:, 1]
    y_proba_rf = tuned_rf_model.predict_proba(X_test)[:, 1]
    y_proba_xgb = tuned_xgb_model.predict_proba(X_test)[:, 1]

    # 2) Compute FPR, TPR, AUC for each
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)

    auc_lr = auc(fpr_lr, tpr_lr)
    auc_rf = auc(fpr_rf, tpr_rf)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    # 3) Build two DataFrames, then concatenate
    df_lr = pd.DataFrame({
        "fpr":   fpr_lr,
        "tpr":   tpr_lr,
        "model": f"Logistic Regression (AUC={auc_lr:.2f})"
    })
    df_rf = pd.DataFrame({
        "fpr":   fpr_rf,
        "tpr":   tpr_rf,
        "model": f"Random Forest (AUC={auc_rf:.2f})"
    })
    df_xgb = pd.DataFrame({
        "fpr":   fpr_xgb,
        "tpr":   tpr_xgb,
        "model": f"XGBoost (AUC={auc_xgb:.2f})"
    })
    df_all = pd.concat([df_lr, df_rf, df_xgb], ignore_index=True)

    # 4) Construct a Plotnine line plot with fixed aspect ratio
    p = (
        ggplot(df_all, aes(x="fpr", y="tpr", color="model"))
        + geom_line(size=1.2)
        + geom_abline(linetype="dashed", slope=1, intercept=0, color="gray")
        + coord_fixed(ratio=1, xlim=(0, 1), ylim=(0, 1))
        + theme_bw()
        + labs(
            title="ROC Curves by Model",
            x="False Positive Rate",
            y="True Positive Rate",
            color="Model"
        )
    )
    # 2) convert it into a Matplotlib Figure
    fig = p.draw()
    # 3) close the ggplot’s internal figure so no double–render
    plt.close(fig)
    return fig


def compare_model_cv(
    model_pipeline: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> pd.DataFrame:
    """
    Run 5-fold stratified CV on a single model pipeline. Return a DataFrame
    with two rows:
      - Model        (the given model_name)
      - Metric       ("Mean Accuracy" and "Runtime (s)")
      - Value        (fold-mean over 5 folds)
      - Error        (fold-std over 5 folds)
    """
    # 1) Set up a stratified 5-fold splitter
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_scores = []
    fold_times  = []

    # 2) Loop over each fold
    for train_idx, test_idx in cv.split(X_train, y_train):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]

        start = time.time()
        model_pipeline.fit(X_tr, y_tr.values.ravel())
        score = model_pipeline.score(X_te, y_te)
        end = time.time()

        fold_scores.append(score)
        fold_times.append(end - start)

    # 3) Compute means and standard deviations
    scores_arr    = np.array(fold_scores)
    times_arr     = np.array(fold_times)
    mean_accuracy = scores_arr.mean()
    std_accuracy  = scores_arr.std()
    mean_runtime  = times_arr.mean()
    std_runtime   = times_arr.std()

    # 4) Build a “long” DataFrame (two rows)
    df_long = pd.DataFrame({
        "Model":  [model_name, model_name],
        "Metric": ["Mean Accuracy", "Runtime (s)"],
        "Value":  [mean_accuracy, mean_runtime],
        "Error":  [std_accuracy, std_runtime]
    })

    # 5) Mark “Model” as categorical (in case you add more later)
    df_long["Model"] = pd.Categorical(
        df_long["Model"],
        categories=[model_name],
        ordered=True
    )

    return df_long


def plot_model_cv(
    cv_stats: pd.DataFrame,
    title: str = "CV Accuracy & Runtime"
) -> Any:
    """
    Take a DataFrame with columns [Model, Metric, Value, Error] (e.g. two rows
    for “Mean Accuracy” and “Runtime”) and return a Plotnine plot faceted by Metric.
    """
    p = (
        ggplot(cv_stats, aes(x="Model", y="Value"))
        + geom_point(size=3)
        + geom_errorbar(
            aes(ymin="Value - Error", ymax="Value + Error"),
            width=0.15
        )
        + facet_wrap("~Metric", scales="free_y")
        + labs(title=title, x="", y="")
        + theme_bw()
        + theme(axis_text_x=element_text(rotation=15, hjust=1))
    )
    fig = p.draw()
    # 3) close the ggplot’s internal figure so no double–render
    plt.close(fig)
    return fig

def merge_cv_stats(
    lr_stats: pd.DataFrame,
    rf_stats: pd.DataFrame,
    xgb_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Take the two single‐model CV‐summary DataFrames (each with columns
    [Model, Metric, Value, Error]) and concatenate them into one long table.
    """
    return pd.concat([lr_stats, rf_stats, xgb_stats], ignore_index=True)