import os
import pickle
from typing import Dict

import dvc.api
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from core import logger
from saver import Saver
from tracking import log_and_register_model_with_mlflow, move_model_to_prod
from utils import authenticate


def train(model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray) -> None:

    if model is None:
        logger.error("Model is None.")
        raise ValueError("Model is None.")
    
    model.fit(X_train, y_train)
    logger.success("Model trained successfully.")


def train_RandomizedSearchCV(
    model: BaseEstimator,
    cfg: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):

    if model is None:
        logger.error("Model is None.")
        raise ValueError("Model is None.")

    params = cfg["tuning"]["random_forest"]
    search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=cfg["tuning"]["n_iter"],
        cv=cfg["tuning"]["cv"],
        verbose=2
    )
    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best score: {search.best_score_}")
    logger.success("Randomized Search CV completed.")

    return search.best_estimator_, search.best_params_


if __name__ == "__main__":
    cfg = dvc.api.params_show()

    train_path = os.path.join(cfg["paths"]["data"]["processed_data"], cfg["names"]["train_data"])
    X_train = pd.read_csv(train_path, sep=",")
    y_train = X_train.pop(cfg["dataset"]["target_col"])

    preproc_path = os.path.join(
        cfg["paths"]["models_parent_dir"],
        cfg["names"]["model_name"],
        f"{cfg['names']['columns_transformer']}.pkl"
    )
    with open(preproc_path, "rb") as f:
        preprocessor = pickle.load(f)

    base_model = RandomForestClassifier(**cfg["hyperparameters"]["random_forest"])
    best_model, best_params = train_RandomizedSearchCV(base_model, cfg, X_train, y_train)

    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", best_model),
    ])

    # Save the model
    model_dir = os.path.join(cfg["paths"]["models_parent_dir"], cfg["names"]["model_name"])
    Saver.save_model(full_pipeline, model_name=cfg["names"]["model_name"], dir=model_dir)

    client: mlflow.client.MlflowClient = authenticate(cfg)
    test_data_path = os.path.join(cfg["paths"]["data"]["interim_data"], cfg["names"]["train_data"])
    test_df = pd.read_csv(test_data_path, sep=",")

    model_details, run_id = log_and_register_model_with_mlflow(
        final_model=full_pipeline,
        test_df=test_df,
        cfg=cfg,
        params=best_params,
    )

    move_model_to_prod(client=client, model_details=model_details)
    logger.info("Training pipeline execution completed.")
