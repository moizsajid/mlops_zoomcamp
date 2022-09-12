import argparse
import os
import pickle

import mlflow
import numpy as np
import xgboost as xg
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("xgb-hyperopt")


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path, num_trials):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):

        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)

            xgb = xg.XGBRegressor(**params)
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {"loss": rmse, "status": STATUS_OK}

    search_space = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 10, 50, 5)),
        "random_state": 0,
    }

    rstate = np.random.default_rng(0)  # for reproducible results

    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved.",
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=50,
        help="the number of parameter evaluations for the optimizer to explore.",
    )
    args = parser.parse_args()

    run(args.data_path, args.max_evals)
