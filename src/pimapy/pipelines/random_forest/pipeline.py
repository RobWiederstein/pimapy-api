from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    train_tuned_rf,
    plot_rf_feature_importance,
)

# nodes used across multiple pipelines
from pimapy.nodes.common import plot_confusion_matrix, compute_rf_metrics

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Node 1: Train & tune a Random Forest
            node(
                func=train_tuned_rf,
                inputs=[
                    "X_train",                    # persisted training features
                    "y_train",                    # persisted training labels
                    "X_test",                     # persisted test features
                    "y_test",                     # persisted test labels
                    "params:random_forest.tuning"  # e.g. param_grid, cv, scoring, random_state
                ],
                outputs=[
                    "tuned_rf_model",     # PickleDataset pointing to data/06_models/tuned_rf_model.pkl
                    "y_test_for_rf",      # MemoryDataset (so downstream nodes can use it)
                    "y_pred_for_rf",      # MemoryDataset
                    "y_proba_for_rf",     # MemoryDataset
                    "best_params_rf",     # MemoryDataset
                    "run_id_rf"           # MemoryDataset
                ],
                name="train_tuned_rf_node",
            ),

            # Node 2: Compute RF metrics (one‐row DataFrame)
            node(
                func=compute_rf_metrics,
                inputs=[
                    "y_test_for_rf",
                    "y_pred_for_rf",
                    "y_proba_for_rf",
                    "params:random_forest.tuning",
                    "best_params_rf",
                    "run_id_rf"
                ],
                outputs="rf_metrics",   # pandas.CSVDataset, e.g. data/07_model_validation/rf_metrics.csv
                name="compute_rf_metrics_node",
            ),

            # Node 3: Plot the RF confusion matrix
            node(
                func=plot_confusion_matrix,
                inputs=["y_test_for_rf", "y_pred_for_rf", "run_id_rf"],
                outputs="rf_confusion_matrix_plot",  # MatplotlibDataset or PickleDataset, e.g. data/08_reporting/plt/rf_cm.png
                name="plot_rf_cm_node",
            ),
            node(
                func=plot_rf_feature_importance,
                inputs=["tuned_rf_model", "X_train"],
                outputs="rf_feature_importance_plot",
                name="plot_rf_feature_importance_node",
            ),
        ]
    )