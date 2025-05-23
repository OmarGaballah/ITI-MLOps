import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from skore import EstimatorReport

from globals import logger


def log_and_register_model_with_mlflow(final_model, test_df, cfg, params):
    """
    Logs and registers a model with MLflow.

    Args:
        final_model: Trained model used for inference.
        X: Features used to infer model signature.
        cfg: Configuration dictionary with model paths and names.
        params: Dictionary of parameters to log.
        ModelWrapper: A custom wrapper class for the model.
        logger: Logger instance for info messages.

    Returns:
        Tuple containing the registered model details and run ID.
    """
    # to remove warning of representing np.nan
    integer_columns = test_df.select_dtypes(include=["int"]).columns
    test_df[integer_columns] = test_df[integer_columns].astype(float)
    X = test_df.drop(columns=cfg["dataset"]["target_col"])
    y = test_df[cfg["dataset"]["target_col"]]
    with mlflow.start_run():
        mlflow.autolog()
        run_id = mlflow.active_run().info.run_id

        # Infer the model signature
        train_preds = final_model.predict(X)
        signature = infer_signature(X, train_preds)

        # Log Model
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path=cfg["paths"]["models_parent_dir"],
            registered_model_name=cfg["names"]["model_name"],
            signature=signature,
            input_example=X.iloc[0:1],
        )

        # Log parameters and metrics
        mlflow.log_params(params)
        final_report = EstimatorReport(final_model, X_test=X, y_test=y)
        metrics = {
            "accuracy": final_report.metrics.accuracy(),
            "precision": final_report.metrics.precision(),
            "recall": final_report.metrics.recall(),
            "roc_auc": final_report.metrics.roc_auc(),
        }
        mlflow.log_metrics(reformat_metrics(metrics))

        # Register the model
        model_uri = f"runs:/{run_id}/{cfg['paths']['models_parent_dir']}"
        model_details = mlflow.register_model(model_uri=model_uri, name=cfg["names"]["model_name"])

        logger.error(model_details)
        logger.info("Model registered successfully!!")

        return model_details, run_id


def reformat_metrics(metrics):
    reformatted_metrics = dict()
    for metric, value in metrics.items():
        if isinstance(value, dict):
            for k, v in value.items():
                reformatted_metrics[f"{metric}_{k}"] = v
        else:
            reformatted_metrics[metric] = value
    return reformatted_metrics


def move_model_to_prod(client: mlflow.client.MlflowClient, model_details) -> None:
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="production",
    )
    # add tag that the model is in production
    client.set_model_version_tag(
        name=model_details.name,
        version=model_details.version,
        key="production",
        value="true",
    )
    logger.info("Model transitioned to prod stage")
