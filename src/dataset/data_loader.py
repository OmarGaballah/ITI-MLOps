import os

import pandas as pd

from core import logger


class Dataset:
    def __init__(
        self,
        data: str | pd.DataFrame,
        target_col: str | None = None,
    ) -> None:
        self.target_col: str = target_col

        # Load data based on the type of `data`
        print(f"Data type: {type(data)}")
        if isinstance(data, str):
            self.filepath: str = data
            self.df: pd.DataFrame = Dataset.load_dataset(filepath=data)
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            self.filepath: str = "DataFrame"
            self.df: pd.DataFrame = data
        else:
            logger.error("Invalid data type. Expected a file path (str) or a DataFrame.")
            raise TypeError("Invalid data type. Expected a file path (str) or a DataFrame.")

    @staticmethod
    def load_dataset(filepath: str) -> pd.DataFrame:
        """
        Read a dataset from a CSV file.
        """
        if not os.path.isfile(filepath):
            logger.error(f"File {filepath} does not exist or is not a file.")
            raise FileNotFoundError(f"File {filepath} does not exist or is not a file.")
        df = pd.read_csv(filepath, sep=",")
        logger.success(f"Dataset loaded from {filepath}.")
        return df

    def split_dataset(
        self, test_ratio: float = 0.2
    ) -> tuple["Dataset", "Dataset", "Dataset", "Dataset"]:
        """
        Split a dataset into training and testing sets.
        """
        # choose random sample from the dataset
        train_df = self.df.sample(frac=1 - test_ratio, random_state=42)
        test_df = self.df.drop(train_df.index)
        X_train = train_df.drop(columns=[self.target_col])
        y_train = train_df[self.target_col]
        X_test = test_df.drop(columns=[self.target_col])
        y_test = test_df[self.target_col]
        logger.success("Dataset split into training and testing sets.")
        return Dataset(X_train), Dataset(y_train), Dataset(X_test), Dataset(y_test)

    def engineer_features(self) -> "Dataset":

        self.df["Title"] = self.df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
        self.df["Title"] = self.df["Title"].replace(
            [
                "Lady",
                "Countess",
                "Capt",
                "Col",
                "Don",
                "Dr",
                "Major",
                "Rev",
                "Sir",
                "Jonkheer",
                "Dona",
            ],
            "Rare",
        )
        self.df["Title"] = self.df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

        self.df["FamilySize"] = self.df["SibSp"] + self.df["Parch"] + 1
        self.df["IsAlone"] = (self.df["FamilySize"] == 1).astype(int)

        self.df["Deck"] = self.df["Cabin"].astype(str).str[0]
        self.df["Deck"] = self.df["Deck"].fillna("U")

        ticket_counts = self.df["Ticket"].value_counts()
        self.df["TicketGroupSize"] = self.df["Ticket"].map(ticket_counts)

        self.df["FarePerPerson"] = self.df["Fare"] / self.df["FamilySize"]
        self.df["FarePerPerson"].replace([np.inf, -np.inf], np.nan, inplace=True)

        self.df["AgeBin"] = pd.cut(
            self.df["Age"], bins=[0, 12, 20, 40, 60, 80], labels=False, include_lowest=True
        )
        self.df["FareBin"] = pd.qcut(self.df["Fare"], 4, labels=False)

        self.df["Pclass*AgeBin"] = self.df["Pclass"] * self.df["AgeBin"].fillna(0).astype(int)
        self.df["Sex_Pclass"] = self.df["Sex"].astype(str) + "_" + self.df["Pclass"].astype(str)

        self.df["CabinMissing"] = self.df["Cabin"].isnull().astype(int)
        self.df["AgeMissing"] = self.df["Age"].isnull().astype(int)
        
        logger.success("Feature engineering completed.")


        return Dataset(self.df, target_col="Survived")

    def get(self) -> pd.DataFrame:
        """
        Get the DataFrame.
        """
        return self.df

    @staticmethod
    def stack(
        vertical: bool = True,
        **kwargs,
    ) -> "Dataset":
        dfs = []
        for name, obj in kwargs.items():
            if isinstance(obj, Dataset):
                dfs.append(obj.get())
            elif isinstance(obj, pd.DataFrame):
                dfs.append(obj)
            else:
                raise TypeError(
                    f"Invalid type for {name}. Expected Dataset or DataFrame, got {type(obj)}."
                )
        axis = 0 if vertical else 1
        concatenated_df = pd.concat(dfs, axis=axis)
        return Dataset(concatenated_df)
