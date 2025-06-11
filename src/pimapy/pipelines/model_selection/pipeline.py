# In src/pimapy/pipelines/model_selection/pipeline.py
from kedro.pipeline import Pipeline, node, pipeline

# 1. Import the individual, modular tuning pipelines.
from pimapy.pipelines.logistic_regression.pipeline import create_pipeline as lr_pipeline
from pimapy.pipelines.random_forest.pipeline import create_pipeline as rf_pipeline
from pimapy.pipelines.xgboost.pipeline import create_pipeline as xgb_pipeline

# 2. Import the common nodes for the selection and evaluation stages.
from pimapy.nodes.common import (
    select_champion_model,
    evaluate_champion_on_test_set,
    plot_confusion_matrix
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the master model selection and evaluation pipeline.
    """

    lr_tuning_pipe = pipeline(
        pipe=lr_pipeline(),
        inputs={"X_train", "y_train"},
        parameters={"params:lr_tuning"},
        namespace="logistic_regression"
    )

    rf_tuning_pipe = pipeline(
        pipe=rf_pipeline(),
        inputs={"X_train", "y_train"},
        parameters={"params:random_forest.tuning"},
        namespace="random_forest"
    )

    xgb_tuning_pipe = pipeline(
        pipe=xgb_pipeline(),
        inputs={"X_train", "y_train"},
        parameters={"params:xgb.tuning_params"},
        namespace="xgboost"
    )

    # The selection and evaluation pipeline remains the same.
    selection_and_evaluation_pipe = pipeline(
        [
            node(
                func=select_champion_model,
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
        ],
        inputs={
            "X_test",
            "y_test",
            "logistic_regression.lr_candidate",
            "random_forest.rf_candidate",
            "xgboost.xgb_candidate"
        }
    )

    # The final combined pipeline structure is unchanged.
    return (
        lr_tuning_pipe
        + rf_tuning_pipe
        + xgb_tuning_pipe
        + selection_and_evaluation_pipe
    )
