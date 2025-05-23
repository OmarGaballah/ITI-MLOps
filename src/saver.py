
import os
import pickle

import numpy as np
import pandas as pd

from dataset.data_loader import Dataset
from dataset.process_data import Preprocessor
from globals import logger


class Saver:
    @staticmethod
    def save_processed_data(
        X: np.ndarray,
        y: np.ndarray,
        target_col: str,
        processor: Preprocessor,
        filename: str,
        dir: str,
    ):
        """
        Save the processed data to a CSV file.
        """
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        filepath = os.path.join(dir, filename)
        processed_df = pd.DataFrame(X, columns=processor.get_feature_names_from_preprocessor())
        processed_df[target_col] = y
        processed_df.to_csv(filepath, index=False)
        logger.success(f"Processed data saved to {filepath}.")

    @staticmethod
    def save_dataset(dataset: Dataset, filename: str, dir: str) -> None:
        """
        Save the dataset to the specified directory.
        """
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        filepath = os.path.join(dir, filename)
        dataset.get().to_csv(filepath, sep=",")
        logger.success(f"Dataset saved to {filepath}.")

    @staticmethod
    def save_model(model, model_name: str, dir: str) -> None:
        """
        Save the model to the specified directory.
        """
        if os.path.exists(dir) is False:
            os.makedirs(dir)
        filepath = os.path.join(dir, f"{model_name}.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

        logger.success(f"Model saved to {filepath}.")
