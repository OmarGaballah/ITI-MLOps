
import json
import os
import pickle
from typing import Dict

import dvc.api
import mlflow
import pandas as pd
from skore import EstimatorReport

from dataset.data_loader import Dataset
from globals import logger
from utils import get_mlflow_client, setup_dagshub


def evaluate(cfg: Dict) -> None:
    logger.info("loading model")
    client = get_mlflow_client()
    logger.error(client.tracking_uri)
    version = client.get_latest_versions(name=cfg['names']["model_name"])[0].version
    logger.error(f"version: {version}")
    final_model = mlflow.sklearn.load_model(model_uri=f"models:/{cfg['names']['model_name']}/{version}")

    data = Dataset(
        data=os.path.join(cfg["paths"]["data"]["interim_data"], cfg["names"]["val_data"]),
        target_col=cfg["dataset"]["target_col"],
    )
    X_test = data.get().drop(columns=cfg["dataset"]["target_col"])
    y_test = data.get()[cfg["dataset"]["target_col"]]

    final_report = EstimatorReport(final_model, X_test=X_test, y_test=y_test)
    logger.info("creating evaluation report")
    evaluation_report = {
        "model_name": cfg["names"]["model_name"],
        "estimator_name": final_report.estimator_name_,
        "fitting_time": final_report.fit_time_,
        "accuracy": final_report.metrics.accuracy(),
        "precision": final_report.metrics.precision(),
        "recall": final_report.metrics.recall(),
        "prediction_time": final_report.metrics.timings(),
    }
    logger.info("saving evaluation report")
    if not os.path.exists(
        os.path.join(cfg["paths"]["reports_parent_dir"], cfg["names"]["model_name"])
    ):
        os.makedirs(os.path.join(cfg["paths"]["reports_parent_dir"], cfg["names"]["model_name"]))
    with open(
        os.path.join(
            cfg["paths"]["reports_parent_dir"],
            cfg["names"]["model_name"],
            "evaluation_report.json",
        ),
        "w",
    ) as js:
        json.dump(evaluation_report, js, indent=4)


def generate_submission_file(cfg: Dict) -> None:
    logger.info("loading model")
    with open(
        os.path.join(
            cfg["paths"]["models_parent_dir"],
            cfg["names"]["model_name"],
            f"{cfg['names']['model_name']}.pkl",
        ),
        "rb",
    ) as pkl:
        final_model = pickle.load(pkl)

    test_data = Dataset(
        data=os.path.join(cfg["paths"]["data"]["raw_data"], cfg["names"]["test_data"]),
    )

    test_id = test_data.engineer_features()
    test_id = test_data.get()[cfg["dataset"]["id_col"]]

    logger.info("creating submission file")
    submission_df = pd.DataFrame()
    submission_df[cfg["dataset"]["id_col"]] = test_id
    submission_df[cfg["dataset"]["target_col"]] = final_model.predict(test_data.get())
    submission_df.to_csv(
        os.path.join(
            cfg["paths"]["models_parent_dir"],
            cfg["names"]["model_name"],
            cfg["names"]["submission_name"],
        ),
        index=False,
    )


if __name__ == "__main__":
    cfg=dvc.api.params_show()
    setup_dagshub(cfg=cfg)
    evaluate(cfg=cfg)
