import os

import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from dataset.data_loader import Dataset
from globals import logger
from modeling.evaluate import evaluate, generate_submission_file
from modeling.train import train_RandomizedSearchCV
from preprocessing import preprocess_train
from saver import Saver


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Pipeline Parameters: \n{OmegaConf.to_yaml(cfg)}")
    logger.info("Training started")
    cfg = cfg["pipeline"]
    train_ds = Dataset(
        data=os.path.join(cfg["paths"]["data"]["raw_data"], cfg["names"]["train_data"]),
        target_col=cfg["dataset"]["target_col"],
    )
    train_ds = train_ds.engineer_features()

    X_train, y_train, X_val, y_val, preprocessor = preprocess_train(
        train_ds,
        cfg=cfg,
    )
    model = RandomForestClassifier(
        **OmegaConf.to_container(cfg["hyperparameters"]["random_forest"], resolve=True),
    )

    model = train_RandomizedSearchCV(model, cfg, X_train, y_train)

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

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor.get_pipeline()),
            ("model", model),
        ]
    )

    Saver.save_model(
        full_pipeline,
        model_name=cfg["names"]["model_name"],
        dir=os.path.join(cfg["paths"]["models_parent_dir"], cfg["names"]["model_name"]),
    )

    logger.info("Training finished")
    evaluate(cfg=cfg)

    generate_submission_file(cfg=cfg)


if __name__ == "__main__":
    main()