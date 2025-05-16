from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)


class Preprocessor:
    def __init__(
        self,
        pipeline_config: Dict[str, str] = None,
    ):
        self.pipeline_config = pipeline_config or {}

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        transformers = []

        transformers.append(("drop_columns", "drop", self.pipeline_config.get("drop", [])))
        transformers.extend(self.__create_encode_steps())
        transformers.extend(self.__create_scaling_steps())
        transformers.extend(self.__create_imputing_steps())

        preprocessor = ColumnTransformer(transformers=transformers)

        self.pipeline = preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() before transform().")
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.pipeline.fit_transform(X)

    def get_pipeline(self) -> ColumnTransformer:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() before get_pipeline().")
        return self.pipeline

    def __create_encode_steps(self) -> List[Tuple[str, BaseEstimator, List[str]]]:
        encode_steps = []
        if self.pipeline_config.get("encoding"):
            for strategy in self.pipeline_config["encoding"].keys():
                if strategy == "onehot":
                    cols = self.pipeline_config["encoding"]["onehot"]
                    encode_steps.append(
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                            ),
                            cols,
                        )
                    )
                elif strategy == "ordinal":
                    cols = self.pipeline_config["encoding"]["ordinal"]
                    encode_steps.append(
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                            cols,
                        )
                    )
                elif strategy == "target":
                    cols = self.pipeline_config["encoding"]["target"]
                    encode_steps.append(("target", TargetEncoder(), cols))
        return encode_steps

    def __create_scaling_steps(self) -> List[Tuple[str, BaseEstimator, List[str]]]:
        scaling_steps = []
        if self.pipeline_config.get("scaling"):
            for strategy in self.pipeline_config["scaling"].keys():
                if strategy == "standard":
                    cols = self.pipeline_config["scaling"]["standard"]
                    scaling_steps.append(("standard", StandardScaler(), cols))
                elif strategy == "minmax":
                    cols = self.pipeline_config["scaling"]["minmax"]
                    scaling_steps.append(("minmax", MinMaxScaler(), cols))
                elif strategy == "robust":
                    cols = self.pipeline_config["scaling"]["robust"]
                    scaling_steps.append(("robust", RobustScaler(), cols))
        return scaling_steps

    def __create_imputing_steps(self) -> List[Tuple[str, BaseEstimator, List[str]]]:
        imputing_steps = []
        if self.pipeline_config.get("imputation"):
            for strategy in self.pipeline_config["imputation"].keys():
                if strategy == "mean":
                    cols = self.pipeline_config["imputation"]["mean"]
                    imputing_steps.append(
                        ("mean", SimpleImputer(strategy="mean", add_indicator=True), cols)
                    )
                elif strategy == "median":
                    cols = self.pipeline_config["imputation"]["median"]
                    imputing_steps.append(
                        ("median", SimpleImputer(strategy="median", add_indicator=True), cols)
                    )
                elif strategy == "most_frequent":
                    cols = self.pipeline_config["imputation"]["most_frequent"]
                    imputing_steps.append(
                        (
                            "most_frequent",
                            SimpleImputer(strategy="most_frequent", add_indicator=True),
                            cols,
                        )
                    )
                elif strategy == "constant":
                    cols_fill: Dict = self.pipeline_config["imputation"]["constant"]
                    for col, fill_value in cols_fill.items():
                        imputing_steps.append(
                            (
                                "constant",
                                SimpleImputer(
                                    strategy="constant", fill_value=fill_value, add_indicator=True
                                ),
                                col,
                            )
                        )
        return imputing_steps

    def get_feature_names_from_preprocessor(self) -> List[str]:
        """
        Extract feature names from a ColumnTransformer after encoding.

        Parameters:
        - preprocessor: The fitted ColumnTransformer object.

        Returns:
        - A list of feature names.
        """
        feature_names = []
        for name, transformer, columns in self.pipeline.transformers_:
            if transformer == "drop" or transformer is None:
                continue  
            elif hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)
        return feature_names