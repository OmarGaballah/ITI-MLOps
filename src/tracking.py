import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.client import MlflowClient
from skore import EstimatorReport

from core import logger


def log_and_register_model_with_mlflow(final_model, test_df, cfg, params):

    int_cols = test_df.select_dtypes(include=["int"]).columns
    test_df[int_cols] = test_df[int_cols].astype(float)

    X = test_df.drop(columns=cfg["dataset"]["target_col"])
    y = test_df[cfg["dataset"]["target_col"]]

    with mlflow.start_run():
        mlflow.autolog()
        run_id = mlflow.active_run().info.run_id

        signature = infer_signature(X, final_model.predict(X))
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path=cfg["paths"]["models_parent_dir"],
            registered_model_name=cfg["names"]["model_name"],
            signature=signature,
            input_example=X.iloc[:1],
        )

        mlflow.log_params(params)
        report = EstimatorReport(final_model, X_test=X, y_test=y)
        metrics = {
            "accuracy": report.metrics.accuracy(),
            "precision": report.metrics.precision(),
            "recall": report.metrics.recall(),
            "roc_auc": report.metrics.roc_auc(),
        }
        mlflow.log_metrics(_flatten_metrics(metrics))

        model_uri = f"runs:/{run_id}/{cfg['paths']['models_parent_dir']}"
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=cfg["names"]["model_name"],
        )

        logger.error(model_details)
        logger.info("Model registered successfully!")

        return model_details, run_id


def _flatten_metrics(metrics: dict) -> dict:

    flat = {}
    for key, val in metrics.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                flat[f"{key}_{sub_key}"] = sub_val
        else:
            flat[key] = val
    return flat


def move_model_to_prod(client: MlflowClient, model_details) -> None:

    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="production"
    )

    client.set_model_version_tag(
        name=model_details.name,
        version=model_details.version,
        key="production",
        value="true"
    )

    logger.info("Model transitioned to production stage.")
