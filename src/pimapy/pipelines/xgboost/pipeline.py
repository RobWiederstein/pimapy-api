# In src/pimapy/pipelines/xgboost/pipeline.py
from kedro.pipeline import Pipeline, node

# Note the import of the NEW candidate function.
# This fixes the NameError because we now import the function we intend to use.
from .nodes import tune_xgboost_candidate

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the modular pipeline for tuning the XGBoost model candidate.
    This pipeline only uses the training data.
    """
    return Pipeline(
        [
            node(
                # Use the new, correctly imported function
                func=tune_xgboost_candidate,
                # The inputs no longer include the test set
                inputs=[
                    "X_train",
                    "y_train",
                    "params:xgb.tuning_params"
                ],
                # The single output is a dictionary containing the results
                # for this candidate model.
                outputs="xgb_candidate",
                name="tune_xgb_candidate_node",
            )
        ]
    )

