import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from core import logger


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
    cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """
    Perform Randomized Search CV on the model.
    """
    if model is None:
        logger.error("Model is None.")
        raise ValueError("Model is None.")

    params = OmegaConf.to_container(cfg["tuning"]["random_forest"], resolve=True)
    n_iter = cfg["tuning"]["n_iter"]
    cv = cfg["tuning"]["cv"]
    search = RandomizedSearchCV(model, params, n_iter=n_iter, cv=cv, verbose=2)
    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best score: {search.best_score_}")

    logger.success("Randomized Search CV completed.")
    return search.best_estimator_