import os
from typing import Dict

import dvc.api
import numpy as np

from dataset.data_loader import Dataset
from dataset.process_data import Preprocessor
from core import logger
from saver import Saver


def preprocess_train(
    train_df: Dataset,
    cfg: Dict = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Preprocessor]:
    # ''' because OmegaConf must be converted to dict to avoid Transofmers erros in hydra but dvc.api doesn't use OmegaConf'''
    # pipeline_config: Dict = OmegaConf.to_container(cfg["pipeline_config"], resolve=True)
    pipeline_config = cfg["pipeline_config"]
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


if __name__ == "__main__":
    cfg = dvc.api.params_show()
    train_ds = Dataset(
        data=os.path.join(cfg["paths"]["data"]["raw_data"], cfg["names"]["train_data"]),
        target_col=cfg["dataset"]["target_col"],
    )
    train_ds = train_ds.engineer_features()
    X_train, y_train, X_val, y_val, preprocessor = preprocess_train(train_ds, cfg)

    Saver.save_model(
        model=preprocessor.get_pipeline(),
        model_name=cfg["names"]["columns_transformer"],
        dir=os.path.join(cfg["paths"]["models_parent_dir"], cfg["names"]["model_name"]),
    )

    Saver.save_processed_data(
        X_train,
        y_train,
        target_col=cfg["dataset"]["target_col"],
        processor=preprocessor,
        filename=cfg["names"]["train_data"],
        dir=cfg["paths"]["data"]["processed_data"],
    )

    Saver.save_processed_data(
        X_val,
        y_val,
        target_col=cfg["dataset"]["target_col"],
        processor=preprocessor,
        filename=cfg["names"]["val_data"],
        dir=cfg["paths"]["data"]["processed_data"],
    )