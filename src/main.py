
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from config import INTERIM_DATA_DIR, MODELS_DIR, PIPELINE_CONFIG, RAW_DATA_DIR
from dataset.data_loader import Dataset
from globals import logger
from modeling.evaluate import evaluate, generate_submission_file
from modeling.train import train
from preprocessing import preprocess_train
from saver import Saver


def main() -> None:
    logger.info("Training started")
    train_ds = Dataset(data=os.path.join(RAW_DATA_DIR, "train.csv"), target_col="Survived")
    train_ds = train_ds.engineer_features()

    Saver.save_dataset(train_ds, filename="train.csv", dir=INTERIM_DATA_DIR)

    X_train, y_train, X_val, y_val, preprocessor = preprocess_train(
        train_ds,
        pipeline_config=PIPELINE_CONFIG,
    )

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

    train(model, X_train, y_train)

    Saver.save_processed_data(
        X_train, y_train, target_col="Survived", processor=preprocessor, filename="train.csv"
    )
    Saver.save_processed_data(
        X_val, y_val, target_col="Survived", processor=preprocessor, filename="val.csv"
    )

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor.get_pipeline()),
            ("model", model),
        ]
    )

    Saver.save_model(
        full_pipeline,
        model_name="random_forest",
        dir=MODELS_DIR / "random_forest",
    )

    logger.info("Training finished")
    evaluate(model_name="random_forest")

    generate_submission_file(model_name="random_forest")


if __name__ == "__main__":
    main()
