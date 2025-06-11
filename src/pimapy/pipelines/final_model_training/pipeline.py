# In src/pimapy/pipelines/final_model_training/pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
import logging

from kedro.pipeline import Pipeline as KedroPipeline, node, pipeline

log = logging.getLogger(__name__)


# --- Pipeline-Specific Node Logic ---
# This function is defined here because it is only used in this pipeline.

def train_final_model(
    champion_model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Pipeline:
    """
    Combines training and test data and fits the champion model on all available data.
    """
    log.info("Combining train and test sets for final model training.")
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    y_all_flat = y_all.values.ravel()
    
    log.info("Fitting the final model on all data.")
    champion_model.fit(X_all, y_all_flat)
    
    return champion_model


# --- Pipeline Definition ---

def create_pipeline(**kwargs) -> KedroPipeline:
    """Creates the pipeline to train the final production model."""
    return pipeline(
        [
            node(
                func=train_final_model,
                inputs={
                    "champion_model": "champion_model", # From model_selection
                    "X_train": "X_train",
                    "y_train": "y_train",
                    "X_test": "X_test",
                    "y_test": "y_test",
                },
                outputs="production_model", # The final artifact for deployment
                name="train_final_model_on_all_data_node",
            )
        ]
    )