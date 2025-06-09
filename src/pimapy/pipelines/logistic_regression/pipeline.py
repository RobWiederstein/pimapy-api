# src/pimapy/pipelines/modeling/pipeline.py
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    train_tuned_log_regress
)

# nodes used across multiple pipelines
from pimapy.nodes.common import plot_confusion_matrix, compute_lr_metrics

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # 1) Train node: returns best_model, y_test, y_pred, y_proba, best_params, run_id
            node(
                func=train_tuned_log_regress,
                inputs=[
                    "X_train",            # from data_processing
                    "y_train",            # from data_processing
                    "X_test",             # from data_processing
                    "y_test",             # from data_processing
                    "params:lr_tuning"    # hyperparameter dict
                ],
                outputs=[
                    "tuned_lr_model",   # fitted Pipeline
                    "y_test_for_eval",  # true labels
                    "y_pred_for_eval",  # predicted labels
                    "y_proba_for_eval", # predicted probabilities
                    "best_params",      # actual best hyperparameters
                    "run_id"            # timestamp string
                ],
                name="train_tuned_log_regression_node",
            ),

            # 2) Metrics node: consumes y_test, y_pred, y_proba, tuning_params, best_params, run_id
            node(
                func=compute_lr_metrics,
                inputs=[
                    "y_test_for_eval",
                    "y_pred_for_eval",
                    "y_proba_for_eval",
                    "params:lr_tuning",
                    "best_params",
                    "run_id"
                ],
                outputs="lr_metrics",  # DataFrame of metrics and metadata
                name="compute_lr_metrics_node",
            ),

            # 3) Plot node: consumes y_test, y_pred, run_id
            node(
                func=plot_confusion_matrix,
                inputs=[
                    "y_test_for_eval",
                    "y_pred_for_eval",
                    "run_id"
                ],
                outputs="lr_cm_fig",   # Matplotlib Figure
                name="plot_lr_confusion_matrix_node",
            ),
        ]
        
    )

