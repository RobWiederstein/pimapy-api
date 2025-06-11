# In a new folder: src/pimapy/pipelines/model_selection/pipeline.py
from kedro.pipeline import Pipeline, node, pipeline

# 1. Import the individual, modular tuning pipelines you just refactored.
# Kedro will find these based on their location in the `src` directory.
from pimapy.pipelines.logistic_regression.pipeline import create_pipeline as lr_pipeline
from pimapy.pipelines.random_forest.pipeline import create_pipeline as rf_pipeline
from pimapy.pipelines.xgboost.pipeline import create_pipeline as xgb_pipeline

# 2. Import the common nodes for selection and final evaluation.
# These nodes are generic and work on any selected model.
from pimapy.nodes.common import (
    select_champion_model,
    evaluate_champion_on_test_set,
    plot_confusion_matrix
)
# Note: Model-specific plots like feature importance are handled separately.

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the master model selection and evaluation pipeline.

    This pipeline orchestrates a "tournament" between models:
    1. Each model is tuned using cross-validation on the training data.
    2. The model with the best cross-validation score is selected as the champion.
    3. The champion model is evaluated ONCE on the held-out test set.
    """

    # 3. Instantiate each model's tuning pipeline.
    #    The .all_outputs_prefixed() method is crucial. It adds a prefix
    #    to the output name in the Data Catalog (e.g., 'candidate' becomes
    #    'lr_candidate'), preventing naming conflicts between the pipelines.
    lr_tuning_pipe = pipeline(
        pipe=lr_pipeline(),
        namespace="logistic_regression" # This prefixes the output as "logistic_regression.lr_candidate"
                                        # which is a clean way to organize.
    )

    rf_tuning_pipe = pipeline(
        pipe=rf_pipeline(),
        namespace="random_forest"
    )

    xgb_tuning_pipe = pipeline(
        pipe=xgb_pipeline(),
        namespace="xgboost"
    )

    # 4. Define the final part of the pipeline: selecting the champion
    #    and evaluating it on the test set.
    selection_and_evaluation_pipe = pipeline(
        [
            node(
                func=select_champion_model,
                # The input is a LIST of the candidate model results
                # from the upstream tuning pipelines.
                inputs=[
                    "logistic_regression.lr_candidate",
                    "random_forest.rf_candidate",
                    "xgboost.xgb_candidate"
                ],
                outputs=["champion_model", "champion_info"],
                name="select_champion_node"
            ),
            node(
                func=evaluate_champion_on_test_set,
                inputs=["champion_model", "champion_info", "X_test", "y_test"],
                outputs=["final_metrics", "y_test_final", "y_pred_final", "run_id_final"],
                name="evaluate_champion_node"
            ),
            node(
                func=plot_confusion_matrix,
                inputs=["y_test_final", "y_pred_final", "run_id_final"],
                outputs="final_confusion_matrix",
                name="plot_final_confusion_matrix_node"
            ),
            # Note on Feature Importance Plots:
            # Since feature importance plots are specific to the model type (e.g., RF or XGB),
            # they are best handled in a separate, downstream reporting pipeline.
            # That pipeline could take 'champion_info' as an input and decide
            # which plotting function to run based on the champion's name.
        ],
        # Explicitly declare inputs that come from outside this specific pipeline
        inputs={
            "X_test",
            "y_test",
            "logistic_regression.lr_candidate",
            "random_forest.rf_candidate",
            "xgboost.xgb_candidate"
        }
    )

    # 5. Combine all the steps into one master pipeline.
    #    This tells Kedro to first run all the tuning pipelines,
    #    then run the selection and evaluation pipeline.
    return (
        lr_tuning_pipe
        + rf_tuning_pipe
        + xgb_tuning_pipe
        + selection_and_evaluation_pipe
    )

