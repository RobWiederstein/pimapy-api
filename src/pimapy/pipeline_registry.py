from typing import Dict
from kedro.pipeline import Pipeline

from pimapy.pipelines.data_processing.pipeline import create_pipeline as create_data_processing
from pimapy.pipelines.logistic_regression.pipeline import create_pipeline as create_logistic_regression
from pimapy.pipelines.random_forest.pipeline import create_pipeline as create_random_forest
from pimapy.pipelines.xgboost.pipeline import create_pipeline as create_xgboost
from pimapy.pipelines.model_comparison.pipeline import create_pipeline as create_model_comparison
from pimapy.pipelines.deployment.pipeline import create_pipeline as create_deployment




def register_pipelines() -> Dict[str, Pipeline]:
    data_processing           = create_data_processing()
    logistic_regression       = create_logistic_regression()
    random_forest             = create_random_forest()
    xgboost                   = create_xgboost()
    model_comparison          = create_model_comparison()
    deployment                = create_deployment()
    
    return {
        "__default__": 
        data_processing 
        + logistic_regression
        + random_forest
        + xgboost
        + model_comparison,
        "data_processing": data_processing,
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
        "xgboost": xgboost,
        "model_comparison": model_comparison,
        "deployment": deployment
    }
