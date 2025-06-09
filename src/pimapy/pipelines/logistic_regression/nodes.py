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
import itertools

def train_tuned_log_regress(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tuning_params: Dict[str, Any]
) -> Tuple[Pipeline, pd.Series, pd.Series, pd.Series, Dict[str, Any], str]:
    """
    Perform CV‐based tuning for Logistic Regression, refit on the full training set,
    evaluate on the test set, and return:
      - best_model: a fitted Pipeline
      - y_test (unchanged)
      - y_pred: predicted class labels for X_test
      - y_proba: predicted probabilities for X_test (positive class)
      - best_params: the actual hyperparameters chosen by GridSearchCV
      - run_id: timestamp string to the minute
    """
    # Generate a timestamp‐based run_id (YYYYMMDDHHMM)
    now = datetime.now()
    run_id = now.strftime("%Y%m%d%H%M")  # e.g. "202506021437"

    # 1) Build the sklearn Pipeline (scaling + logistic)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                random_state=tuning_params["random_state"],
                max_iter=1000
            )),
        ]
    )

    # 2) Extract hyperparameter tuning settings
    param_grid   = tuning_params["param_grid"]
    cv           = tuning_params["cv"]
    scoring      = tuning_params["scoring"]

    # 3) Set up and run GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train.values.ravel())
    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 4) Make predictions on the test set
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    return (
        best_model,
        y_test,
        pd.Series(y_pred, index=y_test.index),
        pd.Series(y_proba, index=y_test.index),
        best_params,
        run_id
    )