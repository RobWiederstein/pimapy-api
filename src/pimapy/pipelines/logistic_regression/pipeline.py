from kedro.pipeline import Pipeline, node, pipeline
from .nodes import tune_log_regress_candidate

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=tune_log_regress_candidate,
                inputs=["X_train", "y_train", "params:lr_tuning"],
                outputs="lr_candidate",
                name="tune_lr_candidate_node",
            )
        ]
    )