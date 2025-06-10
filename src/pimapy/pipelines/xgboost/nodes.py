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

def train_tuned_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tuning_params: Dict[str, Any],
) -> Tuple[
    Pipeline,        # the fitted XGB pipeline
    pd.Series,       # unchanged y_test
    pd.Series,       # y_pred on X_test
    pd.Series,       # y_proba on X_test (positive‐class probability)
    Dict[str, Any],  # best_params from GridSearchCV
    str              # run_id timestamp (YYYYMMDDHHMM)
]:
    """
    Perform CV‐based tuning for an XGBoost classifier, refit on the full training set,
    evaluate on the test set, and return:
      - best_model: a fitted Pipeline containing the XGBClassifier
      - y_test (unchanged)
      - y_pred: predicted class labels for X_test
      - y_proba: predicted probabilities for X_test (positive class)
      - best_params: the actual hyperparameters chosen by GridSearchCV
      - run_id: timestamp string to the minute

    `tuning_params` should contain:
      - "param_grid": the XGB parameter grid
      - "cv": number of folds or a cross‐validation splitter
      - "scoring": scoring metric (e.g. "roc_auc")
      - "random_state": integer seed for reproducibility
    """
    # Generate a timestamp‐based run_id (YYYYMMDDHHMM)
    now = datetime.now()
    run_id = now.strftime("%Y%m%d%H%M")

    # 1) Build a simple XGB pipeline (no scaling if your features are already numeric;
    #    XGBoost is tree‐based, so scaling is not strictly required)
    xgb_pipe = Pipeline(
        [
            (
                "clf",
                XGBClassifier(
                    random_state=tuning_params["random_state"],
                    eval_metric="logloss",
                ),
            )
        ]
    )

    # 2) Extract hyperparameter tuning settings
    param_grid = tuning_params["param_grid"]   # e.g., xgb_param_grid from notebook
    cv = tuning_params["cv"]
    scoring = tuning_params["scoring"]

    # 3) Set up and run GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    # .ravel() ensures y_train is 1D if it's a DataFrame column
    grid_search.fit(X_train, y_train.values.ravel())

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 4) Predict on the test set
    y_pred = best_model.predict(X_test)
    # `predict_proba` returns an array of shape (n_samples, n_classes);
    # we take the positive‐class probability (index 1)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    return (
        best_model,
        y_test,
        pd.Series(y_pred, index=y_test.index),
        pd.Series(y_proba, index=y_test.index),
        best_params,
        run_id,
    )

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