import argparse
import os
import pickle

import mlflow
import xgboost as xg
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "xgb-hyperopt"
EXPERIMENT_NAME = "xgb-best-models"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.xgboost.autolog()

SPACE = {
    "n_estimators": scope.int(hp.quniform("n_estimators", 10, 50, 5)),
    "random_state": 0,
}


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        params = space_eval(SPACE, params)
        xgb = xg.XGBRegressor(n_estimators=100, booster="gbtree", random_state=0)
        xgb.fit(X_train, y_train)

        # evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, xgb.predict(X_val), squared=False)
        mlflow.log_metric("valid_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, xgb.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)


def run(data_path, log_top):

    client = MlflowClient()

    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.rmse ASC"],
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"],
    )[0]

    # register the best model
    mlflow.register_model(
        model_uri=f"runs:/{best_run.info.run_id}/model", name="price-regressor"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed car price prediciton data was saved.",
    )
    parser.add_argument(
        "--top_n",
        default=5,
        type=int,
        help="the top 'top_n' models will be evaluated to decide which model to promote.",
    )
    args = parser.parse_args()

    run(args.data_path, args.top_n)
