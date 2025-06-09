from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    train_tuned_xgb,
    plot_xgb_feature_importance
)
# nodes used across multiple pipelines
from pimapy.nodes.common import plot_confusion_matrix, compute_xgb_metrics

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_tuned_xgb,
                inputs=[
                    "X_train",                 # 1st input
                    "y_train",                 # 2nd input
                    "X_test",                  # 3rd input
                    "y_test",                  # 4th input
                    "params:xgb.tuning_params" # 5th input: comes from parameters.yml
                ],
                outputs=[
                    "tuned_xgb_model",       # 1st output
                    "y_test_xgb",          # 2nd (unchanged)
                    "y_pred_xgb",      # 3rd
                    "y_proba_xgb",     # 4th
                    "xgb_best_params", # 5th
                    "xgb_run_id",      # 6th
                ],
                name="train_tuned_xgb_node",
            ),
            node(
                func=compute_xgb_metrics,
                inputs=[
                    "y_test_xgb",             # pulled from previous node’s outputs
                    "y_pred_xgb",             # predicted labels
                    "y_proba_xgb",            # predicted probabilities
                    "params:xgb.tuning_params",# same tuning params dict
                    "xgb_best_params",        # best_params from previous node
                    "xgb_run_id"              # run_id from previous node
                ],
                outputs="xgb_metrics",       # single‐row DataFrame of metrics
                name="compute_xgb_metrics_node",
            ),
            node(
                func=plot_confusion_matrix,
                inputs=[
                    "y_pred_xgb",
                    "y_test_xgb",
                    "xgb_run_id" 
                ],
                outputs="xgb_confusion_matrix",  # a Matplotlib figure
                name="plot_xgb_confusion_matrix_node",
            ),
            node(
                func=plot_xgb_feature_importance,
                inputs=["tuned_xgb_model", "X_train", "xgb_run_id"],
                outputs="xgb_feature_importance",
                name="plot_xgb_feature_importance_node",
            ),
        ]
    )