# In src/pimapy/pipelines/model_comparison/pipeline.py

from kedro.pipeline import Pipeline, node

from .nodes import plot_multi_model_roc_from_candidates, plot_cv_scores_from_candidates

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a pipeline to generate comparison plots for all model candidates.
    This pipeline should be run AFTER the main model selection pipeline.
    """
    return Pipeline(
        [
            node(
                func=plot_multi_model_roc_from_candidates,
                # This node takes the list of candidate dicts AND the test set
                inputs=[
                    "logistic_regression.lr_candidate",
                    "random_forest.rf_candidate",
                    "xgboost.xgb_candidate",
                    "X_test",
                    "y_test"
                ],
                outputs="comparison_roc_plot",
                name="plot_comparison_roc_node"
            ),
            node(
                func=plot_cv_scores_from_candidates,
                # This node ONLY needs the candidate dicts
                inputs=[
                    "logistic_regression.lr_candidate",
                    "random_forest.rf_candidate",
                    "xgboost.xgb_candidate",
                ],
                outputs="comparison_cv_scores_plot",
                name="plot_comparison_cv_scores_node"
            ),
        ]
    )

