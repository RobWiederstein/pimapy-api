from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import itertools

from plotnine import (
    ggplot, aes, geom_col, coord_flip,
    labs, theme_bw, geom_segment, geom_point
)
def train_tuned_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tuning_params: Dict[str, Any],
) -> Tuple[
    Pipeline,        # the fitted RF pipeline
    pd.Series,       # unchanged y_test
    pd.Series,       # y_pred on X_test
    pd.Series,       # y_proba on X_test (positive‐class probability)
    Dict[str, Any],  # best_params from GridSearchCV
    str              # run_id timestamp (YYYYMMDDHHMM)
]:
    """
    Perform CV‐based tuning for a Random Forest classifier, refit on the full training set,
    evaluate on the test set, and return:
      - best_model: a fitted Pipeline containing the RF
      - y_test (unchanged)
      - y_pred: predicted class labels for X_test
      - y_proba: predicted probabilities for X_test (positive class)
      - best_params: the actual hyperparameters chosen by GridSearchCV
      - run_id: timestamp string to the minute

    tuning_params should contain:
      - "param_grid": the RF parameter grid
      - "cv": number of folds or a cross‐validation splitter
      - "scoring": scoring metric (e.g. "roc_auc")
      - "random_state": integer seed for reproducibility
    """
    # Generate a timestamp-based run_id (YYYYMMDDHHMM)
    now = datetime.now()
    run_id = now.strftime("%Y%m%d%H%M")

    # 1) Build a simple RF pipeline (no scaling needed for RF)
    pipe = Pipeline(
        [
            (
                "clf",
                RandomForestClassifier(
                    random_state=tuning_params["random_state"],
                    class_weight="balanced"  # mirror class‐weight tuning
                ),
            )
        ]
    )

    # 2) Extract hyperparameter tuning settings
    param_grid = tuning_params["param_grid"]   # e.g. rf_param_grid from notebook
    cv = tuning_params["cv"]
    scoring = tuning_params["scoring"]

    # 3) Set up and run GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
    )
    # .ravel() ensures y_train is 1D
    grid_search.fit(X_train, y_train.values.ravel())

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 4) Predict on the test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    return (
        best_model,
        y_test,
        pd.Series(y_pred, index=y_test.index),
        pd.Series(y_proba, index=y_test.index),
        best_params,
        run_id,
    )

# def compute_rf_metrics(
#     y_test: pd.Series,
#     y_pred: pd.Series,
#     y_proba: pd.Series,
#     tuning_params: Dict[str, Any],
#     best_params: Dict[str, Any],
#     run_id: str
# ) -> pd.DataFrame:
#     """
#     Given true/predicted labels & probabilities, compute a suite of RF metrics.
#     Returns a one‐row DataFrame containing:
#       - run_id, Model="Random Forest"
#       - accuracy, roc_auc, precision, recall, f1, mcc, bal_accuracy
#       - best_params (stringified)
#       - cv_folds, scoring, random_state, stratify_flag
#     """
#     acc = accuracy_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_proba)
#     prec = precision_score(y_test, y_pred, zero_division=0)
#     rec = recall_score(y_test, y_pred, zero_division=0)
#     f1 = f1_score(y_test, y_pred, zero_division=0)
#     mcc = matthews_corrcoef(y_test, y_pred)
#     bal_acc = balanced_accuracy_score(y_test, y_pred)

#     cv_folds = tuning_params["cv"]
#     scoring = tuning_params["scoring"]
#     random_state = tuning_params["random_state"]
#     stratify_flag = tuning_params.get("stratify", True)

#     record = {
#         "run_id":       run_id,
#         "Model":        "Random Forest",
#         "accuracy":     acc,
#         "roc_auc":      roc_auc,
#         "precision":    prec,
#         "recall":       rec,
#         "f1_score":     f1,
#         "matthews_cc":  mcc,
#         "bal_accuracy": bal_acc,
#         "best_params":  str(best_params),
#         "cv_folds":     cv_folds,
#         "scoring":      scoring,
#         "random_state": random_state,
#         "stratify":     stratify_flag
#     }
#     return pd.DataFrame([record])

def plot_rf_feature_importance(
    rf_pipeline: Pipeline,
    X_train: pd.DataFrame
) -> "ggplot":
    """
    Given a fitted RandomForest pipeline (with step "clf") and the original
    training DataFrame, build a horizontally‐oriented lollipop chart of feature importances,
    omitting 'flag_imp' and ordering so the highest importance is at the top.
    """
    # 1) Extract importances
    rf_model    = rf_pipeline.named_steps["clf"]
    importances = rf_model.feature_importances_
    
    # 2) Build a DataFrame mapping feature → importance
    feature_names = list(X_train.columns)
    df_imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances
    })
    
    # 3) Omit 'flag_imp' row if present
    df_imp = df_imp[df_imp["feature"] != "flag_imp"].copy()
    
    # 4) Sort in descending order
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    
    # 5) Reverse the ordering so that highest importance appears at the top after flipping
    ordered_feats = df_imp["feature"].tolist()[::-1]
    df_imp["feature"] = pd.Categorical(
        df_imp["feature"],
        categories=ordered_feats,
        ordered=True
    )
    
    # 6) Build a horizontal lollipop plot by adding coord_flip()
    p = (
        ggplot(df_imp, aes(x="feature", y="importance"))
        + geom_segment(
            aes(x="feature", xend="feature", y=0, yend="importance"),
            color="gray"
        )
        + geom_point(color="steelblue", size=3)
        + coord_flip()                # Rotate so features run vertically on the y-axis
        + theme_bw()
        + labs(
            title="Random Forest Feature Importances (Lollipop)",
            x="Feature",
            y="Importance (mean decrease in impurity)"
        )
    )
    return p