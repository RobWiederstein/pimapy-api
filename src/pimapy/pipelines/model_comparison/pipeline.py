# src/<your_package>/pipelines/model_comparison/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import plot_multi_models_roc, compare_model_cv, plot_model_cv, merge_cv_stats

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=plot_multi_models_roc,
                inputs=[
                    "tuned_lr_model",
                    "tuned_rf_model",
                    "tuned_xgb_model", 
                    "X_test",        
                    "y_test"         
                ],
                outputs="models_roc_plot",  
                name="plot_multi_models_roc_node"
            ),
            node(
                func=compare_model_cv,
                inputs=[
                    "tuned_lr_model",                    # a fitted sklearn Pipeline
                    "params:model_comparison.lr_model_name",# e.g. "Logistic Regression"
                    "X_train",                           # persisted training features
                    "y_train"                            # persisted training labels
                ],
                outputs="lr_cv_stats",               # DataFrame with two rows
                name="compare_lr_cv_node",
            ),

            # 2) CV on tuned RF → writes "rf_cv_stats"
            node(
                func=compare_model_cv,
                inputs=[
                    "tuned_rf_model",
                    "params:model_comparison.rf_model_name",
                    "X_train",
                    "y_train"
                ],
                outputs="rf_cv_stats",
                name="compare_rf_cv_node",
            ),
            # CV on tuned XGB → writes "xgb_cv_stats"
            node(
                func=compare_model_cv,
                inputs=[
                    "tuned_xgb_model", 
                    "params:model_comparison.xgb_model_name",
                    "X_train", 
                    "y_train"
                ],
                outputs="xgb_cv_stats",
                name="compare_xgb_cv_node",
            ),

            # 3) Merge the two into "model_cv_stats"
            node(
                func=merge_cv_stats,
                inputs=["lr_cv_stats", "rf_cv_stats", "xgb_cv_stats"],
                outputs="model_cv_stats",
                name="merge_cv_stats_node",
            ),

            # 4) Plot that merged table → writes "model_cv_plot"
            node(
                func=plot_model_cv,
                inputs="model_cv_stats",
                outputs="model_cv_plot",
                name="plot_model_cv_node",
            ),
        ]
    )
