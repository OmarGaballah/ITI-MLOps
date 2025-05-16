import os
import pickle
from typing import Dict

import dvc.api
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from core import logger
from saver import Saver


def train(model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray) -> None:
    """
    Train the model.
    """
    if model is None:
        logger.error("Model is None.")
        raise ValueError("Model is None.")
    model.fit(X_train, y_train)
    logger.success("Model trained.")


def train_RandomizedSearchCV(
    model: BaseEstimator,
    cfg: Dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """
    Perform Randomized Search CV on the model.
    """
    if model is None:
        logger.error("Model is None.")
        raise ValueError("Model is None.")

    params = cfg["tuning"]["random_forest"]
    n_iter = cfg["tuning"]["n_iter"]
    cv = cfg["tuning"]["cv"]
    search = RandomizedSearchCV(model, params, n_iter=n_iter, cv=cv, verbose=2)
    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best score: {search.best_score_}")

    logger.success("Randomized Search CV completed.")
    return search.best_estimator_


if __name__ == "__main__":
    cfg = dvc.api.params_show()
    X_train = pd.read_csv(
        os.path.join(cfg["paths"]["data"]["processed_data"], cfg["names"]["train_data"]), sep=","
    )
    y_train = X_train.pop(cfg["dataset"]["target_col"])

    # load column transofrmer
    path = os.path.join(
        cfg["paths"]["models_parent_dir"],
        cfg["names"]["model_name"],
        f"{cfg['names']['columns_transformer']}.pkl",
    )
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)

    model = RandomForestClassifier(**cfg["hyperparameters"]["random_forest"])

    model = train_RandomizedSearchCV(model, cfg, X_train, y_train)

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    Saver.save_model(
        full_pipeline,
        model_name=cfg["names"]["model_name"],
        dir=os.path.join(cfg["paths"]["models_parent_dir"], cfg["names"]["model_name"]),
    )

    logger.info("Training finished")