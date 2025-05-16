
import json
import os
import pickle

import pandas as pd
from skore import EstimatorReport

from dataset.data_loader import Dataset
from core import logger, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PIPELINE_CONFIG,MODELS_DIR,REPORTS_DIR,RAW_DATA_DIR



def evaluate(model_name: str) -> None:
    logger.info("loading model")
    with open(os.path.join(MODELS_DIR, model_name, f"{model_name}.pkl"), "rb") as pkl:
        final_model = pickle.load(pkl)

    data = Dataset(
        data=os.path.join(INTERIM_DATA_DIR, "val.csv"),
        target_col="Survived",
    )
    X_test = data.get().drop(columns=["Survived"])
    y_test = data.get()["Survived"]

    final_report = EstimatorReport(final_model, X_test=X_test, y_test=y_test)
    logger.info("creating evaluation report")
    evaluation_report = {
        "model_name": model_name,
        "estimator_name": final_report.estimator_name_,
        "fitting_time": final_report.fit_time_,
        "accuracy": final_report.metrics.accuracy(),
        "precision": final_report.metrics.precision(),
        "recall": final_report.metrics.recall(),
        "prediction_time": final_report.metrics.timings(),
    }
    logger.info("saving evaluation report")
    if not os.path.exists(os.path.join(REPORTS_DIR, model_name)):
        os.makedirs(os.path.join(REPORTS_DIR, model_name))
    with open(os.path.join(REPORTS_DIR, model_name, "evaluation_report.json"), "w") as js:
        json.dump(evaluation_report, js, indent=4)


def generate_submission_file(model_name: str) -> None:
    logger.info("loading model")
    with open(os.path.join(MODELS_DIR, model_name, f"{model_name}.pkl"), "rb") as pkl:
        final_model = pickle.load(pkl)

    test_data = Dataset(
        data=os.path.join(RAW_DATA_DIR, "test.csv"),
    )

    test_id = test_data.engineer_features()
    test_id = test_data.get()["PassengerId"]

    logger.info("creating submission file")
    submission_df = pd.DataFrame()
    submission_df["PassengerId"] = test_id
    submission_df["Survived"] = final_model.predict(test_data.get())
    submission_df.to_csv(os.path.join(MODELS_DIR, model_name, "submission.csv"), index=False)