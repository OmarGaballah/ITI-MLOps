from typing import Dict

import numpy as np
from omegaconf import DictConfig, OmegaConf

from dataset.data_loader import Dataset
from dataset.process_data import Preprocessor
from globals import logger
from saver import Saver


def preprocess_train(
    train_df: Dataset,
    cfg: DictConfig = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Preprocessor]:
    
    pipeline_config:Dict = OmegaConf.to_container(cfg["pipeline_config"], resolve=True)
    preprocessor = Preprocessor(pipeline_config=pipeline_config)

    X_train, y_train, X_val, y_val = train_df.split_dataset(test_ratio=cfg["dataset"]["test_size"])

    Saver.save_dataset(
        Dataset.stack(vertical=False, df1=X_train, df2=y_train),
        filename=cfg["names"]["train_data"],
        dir=cfg["paths"]["data"]["interim_data"],
    )
    Saver.save_dataset(
        Dataset.stack(vertical=False, df1=X_val, df2=y_val),
        filename=cfg["names"]["val_data"],
        dir=cfg["paths"]["data"]["interim_data"],
    )

    X_train_processed = preprocessor.fit_transform(X_train.get())
    x_val_processed = preprocessor.transform(X_val.get())

    logger.info(f"X_train shape: {X_train_processed.shape}")
    logger.info(f"X_val shape: {x_val_processed.shape}")
    logger.info(f"y_train shape: {y_train.get().shape}")
    logger.info(f"y_val shape: {y_val.get().shape}")
    logger.success("Preprocessing train completed.")

    return (
        X_train_processed,
        np.array(y_train.get()),
        x_val_processed,
        np.array(y_val.get()),
        preprocessor,
    )