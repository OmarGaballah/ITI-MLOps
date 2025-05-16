
import numpy as np

from core import logger,INTERIM_DATA_DIR, PIPELINE_CONFIG
from dataset.data_loader import Dataset
from dataset.process_data import Preprocessor
from saver import Saver


def preprocess_train(
    train_df: Dataset,
    pipeline_config: dict = PIPELINE_CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Preprocessor]:
    preprocessor = Preprocessor(pipeline_config)

    X_train, y_train, X_val, y_val = train_df.split_dataset()

    Saver.save_dataset(
        Dataset.stack(vertical=False, df1=X_train, df2=y_train),
        filename="train.csv",
        dir=INTERIM_DATA_DIR,
    )
    Saver.save_dataset(
        Dataset.stack(vertical=False, df1=X_val, df2=y_val),
        filename="val.csv",
        dir=INTERIM_DATA_DIR,
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