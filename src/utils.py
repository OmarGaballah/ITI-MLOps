import os
import mlflow
import dagshub
from dotenv import load_dotenv

from core import logger


class MLFlowClientSingleton:

    _instance = None
    _client = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MLFlowClientSingleton, cls).__new__(cls)
        return cls._instance


def get_mlflow_client() -> mlflow.client.MlflowClient:
    load_dotenv()
    tracking_uri = os.getenv("TRACKING_URI")
    if MLFlowClientSingleton._client is None:
        MLFlowClientSingleton._client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
        logger.info("MLflow client initialized.")
    return MLFlowClientSingleton._client


def setup_dagshub(cfg: dict) -> None:

    load_dotenv()
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name=os.getenv("DAGSHUB_REPO_NAME"),
        mlflow=cfg["flags"]["use_mlflow"],
    )
    logger.info("Dagshub tracking initialized.")


def authenticate(cfg: dict) -> mlflow.client.MlflowClient:

    setup_dagshub(cfg)
    return get_mlflow_client()
