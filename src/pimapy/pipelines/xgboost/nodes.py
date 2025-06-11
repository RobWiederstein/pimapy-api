from datetime import datetime
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

def tune_xgboost_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuning_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Performs CV-based tuning for an XGBoost classifier on the TRAINING DATA ONLY.

    Returns a dict containing:
      - model_name
      - best_cv_score
      - model_object   (the full GridSearchCV)
      - best_params
      - cv_scores      (list of CV fold scores for the best setting)
    """
    # 1) Build XGB pipeline
    xgb_pipe = Pipeline([
        (
            "clf",
            XGBClassifier(
                random_state=tuning_params["random_state"],
                eval_metric="logloss"
            )
        )
    ])

    # 2) Run GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_pipe,
        param_grid=tuning_params["param_grid"],
        cv=tuning_params["cv"],
        scoring=tuning_params["scoring"],
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    grid_search.fit(X_train, y_train.values.ravel())

    # 3) Extract per-fold test scores at the best index
    best_idx   = grid_search.best_index_
    results    = grid_search.cv_results_
    split_keys = sorted(
        [k for k in results.keys()
         if k.startswith("split") and k.endswith("_test_score")],
        key=lambda s: int(s.split("split")[1].split("_")[0])
    )
    cv_scores = [results[k][best_idx] for k in split_keys]

    # 4) Return the full GridSearchCV plus fold scores
    return {
        "model_name":    "XGBoost",
        "best_cv_score": grid_search.best_score_,
        "model_object":  grid_search,        # full GridSearchCV
        "best_params":   grid_search.best_params_,
        "cv_scores":     cv_scores
    }

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools

def plot_xgb_feature_importance(
    tuned_xgb_model: Pipeline,    # the trained Pipeline (GridSearchCV.best_estimator_)
    X_train: pd.DataFrame,  # original training DataFrame (to get column names)
    run_id: str
) -> plt.Figure:
    """
    Given a fitted XGBoost pipeline and the original X_train DataFrame, plot the
    feature importances as a bar chart. Returns a Matplotlib Figure (dpi=300).
    """
    # 1) Extract the XGBClassifier from the Pipeline
    #    (assumes your Pipeline is: [("scaler", ...), ("clf", XGBClassifier(...))])
    try:
        clf: XGBClassifier = tuned_xgb_model.named_steps["clf"]
    except KeyError:
        raise KeyError(
            "Expected the pipeline to have a step named 'clf' with an XGBClassifier."
        )

    # 2) Get the raw feature importances (a NumPy array)
    #    XGBClassifier stores them under .feature_importances_
    importances = clf.feature_importances_

    # 3) Match importances to column names
    feature_names = list(X_train.columns)
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Number of features ({len(feature_names)}) "
            f"does not match importances length ({len(importances)})."
        )

    # 4) Build a DataFrame and sort by importance (descending)
    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    # 5) Plot as a horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.barh(
        imp_df["feature"],
        imp_df["importance"],
        color="steelblue",
    )
    ax.invert_yaxis()  # highest importance at top
    ax.set_xlabel("Importance")
    ax.set_title(f"XGBoost Feature Importances (run {run_id})")
    fig.tight_layout()

    return fig