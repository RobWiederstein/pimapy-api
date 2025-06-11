from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import itertools

from plotnine import (
    ggplot, aes, geom_col, coord_flip,
    labs, theme_bw, geom_segment, geom_point
)

def tune_random_forest_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuning_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs CV-based tuning for a Random Forest classifier on the TRAINING DATA ONLY.

    Returns a dict containing:
      - model_name
      - best_cv_score
      - model_object   (the full GridSearchCV)
      - best_params
      - cv_scores      (list of CV fold scores for the best setting)
    """
    # 1) Build the RF pipeline
    pipe = Pipeline([
        (
            "clf",
            RandomForestClassifier(
                random_state=tuning_params["random_state"],
                class_weight="balanced"
            ),
        )
    ])

    # 2) Run GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=tuning_params["param_grid"],
        cv=tuning_params["cv"],
        scoring=tuning_params["scoring"],
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train.values.ravel())

    # 3) Extract per-fold test scores at the best index
    best_idx = grid_search.best_index_
    results  = grid_search.cv_results_
    split_keys = sorted(
        [k for k in results.keys() 
         if k.startswith("split") and k.endswith("_test_score")],
        key=lambda s: int(s.split("split")[1].split("_")[0])
    )
    cv_scores = [results[k][best_idx] for k in split_keys]

    # 4) Return everything in one dict
    return {
        "model_name":    "Random Forest",
        "best_cv_score": grid_search.best_score_,
        "model_object":  grid_search,          # full GridSearchCV
        "best_params":   grid_search.best_params_,
        "cv_scores":     cv_scores
    }

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