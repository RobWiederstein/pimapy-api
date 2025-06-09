from typing import Any
import pandas as pd
from sklearn.pipeline import Pipeline
from kedro.pipeline import node, Pipeline as KedroPipeline

def score_model(
    new_data: pd.DataFrame,
    model: Pipeline
) -> pd.DataFrame:
    """
    Loads a saved sklearn Pipeline, runs predict() and predict_proba(),
    and returns a DataFrame with both label and probability.
    """
    preds = model.predict(new_data)
    probs = model.predict_proba(new_data)[:, 1]
    return pd.DataFrame({
        "prediction": preds,
        "p_positive": probs
    }, index=new_data.index)

def create_pipeline(**kwargs) -> KedroPipeline:
    return KedroPipeline(
        [
            node(
                func=score_model,
                inputs=["new_data", "tuned_lr_model"],
                outputs="lr_predictions",
                name="lr_scoring_node",
            )
        ]
    )
