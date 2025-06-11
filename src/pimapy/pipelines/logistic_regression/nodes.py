# In src/pimapy/pipelines/logistic_regression/nodes.py

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def tune_log_regress_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuning_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs CV‐based tuning for Logistic Regression on the TRAINING DATA ONLY.

    Returns a dict containing:
      - model_name
      - best_cv_score
      - model_object   (the full GridSearchCV, so you can still .predict_proba)
      - best_params
      - cv_scores      (list of fold scores for the best param set)
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            random_state=tuning_params["random_state"],
            max_iter=1000
        )),
    ])

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=tuning_params["param_grid"],
        cv=tuning_params["cv"],
        scoring=tuning_params["scoring"],
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train.values.ravel())

    # === extract the per‐fold test scores for the best index ===
    best_i   = grid_search.best_index_
    cv_res   = grid_search.cv_results_
    # split0_test_score, split1_test_score, ...
    split_keys = sorted(
        [k for k in cv_res if k.startswith("split") and k.endswith("_test_score")],
        key=lambda s: int(s.split("split")[1].split("_")[0])
    )
    cv_scores = [cv_res[k][best_i] for k in split_keys]

    candidate_results: Dict[str, Any] = {
        "model_name":    "Logistic Regression",
        "best_cv_score": grid_search.best_score_,
        "model_object":  grid_search,        # keep the full GridSearchCV
        "best_params":   grid_search.best_params_,
        "cv_scores":     cv_scores,          # now defined!
    }

    return candidate_results
