# In src/pimapy/pipeline_registry.py

from typing import Dict
from kedro.pipeline import Pipeline, pipeline

# Import all the necessary pipeline creation functions
from pimapy.pipelines.data_processing.pipeline import create_pipeline as create_data_processing
from pimapy.pipelines.deployment.pipeline import create_pipeline as create_deployment

# Import the NEW master model selection pipeline
from pimapy.pipelines.model_selection.pipeline import create_pipeline as create_model_selection

# We still import the individual model pipelines, but they will be used
# for modular runs, not chained in the default pipeline.
from pimapy.pipelines.logistic_regression.pipeline import create_pipeline as create_logistic_regression
from pimapy.pipelines.random_forest.pipeline import create_pipeline as create_random_forest
from pimapy.pipelines.xgboost.pipeline import create_pipeline as create_xgboost
from pimapy.pipelines.model_comparison.pipeline import create_pipeline as create_model_comparison
from pimapy.pipelines.final_model_training.pipeline import create_pipeline as create_final_model_training

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""

    # Instantiate all the pipelines
    data_processing_pipe = create_data_processing()
    deployment_pipe = create_deployment()
    model_selection_pipe = create_model_selection()
    lr_tuning_pipe = create_logistic_regression()
    rf_tuning_pipe = create_random_forest()
    xgb_tuning_pipe = create_xgboost()
    model_comparison_pipe = create_model_comparison()
    final_model_training_pipe = create_final_model_training()

    return {
        # The __default__ pipeline is now the main, correct, end-to-end workflow:
        # 1. Process the data.
        # 2. Run the model selection tournament and final evaluation.
        "__default__": (data_processing_pipe 
        + model_selection_pipe
        + model_comparison_pipe
        ),

        # You can still run each part of the process individually.
        "data_processing": data_processing_pipe,
        "deployment": deployment_pipe,

        # This is the master pipeline for the entire modeling "tournament".
        # You can run it on its own if the data is already processed.
        "model_selection": model_selection_pipe,
        # The final model training pipeline is now separate.
        "final_model_training": final_model_training_pipe,

        # These allow you to run just the tuning for a single model if needed.
        "tune_lr": lr_tuning_pipe,
        "tune_rf": rf_tuning_pipe,
        "tune_xgb": xgb_tuning_pipe,
        "model_comparison": model_comparison_pipe,
    }
