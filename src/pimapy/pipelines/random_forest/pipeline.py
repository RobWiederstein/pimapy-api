# In src/pimapy/pipelines/random_forest/pipeline.py
from kedro.pipeline import Pipeline, node

# Note the import of the NEW candidate function
from .nodes import tune_random_forest_candidate

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the modular pipeline for tuning the Random Forest model candidate.
    This pipeline only uses the training data.
    """
    return Pipeline(
        [
            node(
                func=tune_random_forest_candidate,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:random_forest.tuning"
                ],
                # The single output is a dictionary containing the results
                # for this candidate model.
                outputs="rf_candidate",
                name="tune_rf_candidate_node",
            )
        ]
    )
