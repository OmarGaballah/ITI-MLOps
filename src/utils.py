import os

import dagshub
from dotenv import load_dotenv
import mlflow

from globals import logger


# implement a singletone class for mlflow client
class MLFlowClientSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MLFlowClientSingleton, cls).__new__(cls)
        return cls._instance


def get_mlflow_client() -> mlflow.client.MlflowClient:
    """
    Get the MLFlow client instance.

    Args:
        tracking_uri (str): The tracking URI for MLFlow.

    Returns:
        mlflow.client.MlflowClient: The MLFlow client instance.
    """
    load_dotenv()
    tracking_uri = os.getenv("TRACKING_URI")
    if not hasattr(MLFlowClientSingleton, "_client"):
        MLFlowClientSingleton._client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    return MLFlowClientSingleton._client


# def setup_mlflow(tracking_uri: str) -> mlflow.client.MlflowClient:
#     mlflow.set_tracking_uri(tracking_uri)
#     client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
#     logger.info("MLFlow Client Defined and tracking URI Setted Successfully.")
#     return client


def setup_dagshub(cfg) -> None:
    load_dotenv()
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name=os.getenv("DAGSHUB_REPO_NAME"),
        mlflow=cfg["flags"]["use_mlflow"],
    )


def authenticate(cfg) -> mlflow.client.MlflowClient:
    """
    Authenticate with MLflow and DagsHub.
    """
    setup_dagshub(cfg)
    client = get_mlflow_client()
    return client
