import os
import pickle

import mlflow
import pandas as pd
from flask import Flask, jsonify, request

RUN_ID = os.getenv("RUN_ID")

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

logged_model = f"s3://mlflow-models-mlopszoomcamp/1/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def prepare_features(car):
    df = pd.DataFrame.from_dict(car, orient="index")
    df = df.T
    df.Levy = df.Levy.str.replace("-", "0")
    df.Levy = df.Levy.astype("int64")
    df.Mileage = df.Mileage.apply(lambda x: str(x).replace("km", " "))
    df.Mileage = df.Mileage.apply(lambda x: int(x))
    df.Doors = df.Doors.str.replace("04-May", "4-5").replace("02-Mar", "2-3")
    df["Engine volume"] = (
        df["Engine volume"].str.replace(r"([a-z," ",A-Z])", "").astype("f")
    )
    df.Airbags = df.Airbags.astype("O")
    df["Cylinders"] = df["Cylinders"].astype("O")
    df["Prod. year"] = df["Prod. year"].astype("O")
    features = df.copy()

    yrs = list(features["Prod. year"].sort_values().unique())
    mapuni = {j: i for i, j in enumerate(yrs)}

    be = load_pickle("output/be.pkl")
    features = be.transform(features)

    features["Prod. year"] = features["Prod. year"].replace(mapuni)
    features["Doors"] = features["Doors"].replace({"2-3": 0, "4-5": 1, ">5": 2})
    features["Leather interior"] = features["Leather interior"].replace(
        {"Yes": 1, "No": 0}
    )
    features["Wheel"] = features["Wheel"].replace(
        {"Left wheel": 1, "Right-hand drive": 0}
    )

    features["Cylinders"] = features["Cylinders"].astype("i")
    features["Airbags"] = features["Airbags"].astype("i")

    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask("price-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    car_information = request.get_json()

    features = prepare_features(car_information)
    pred = predict(features)

    result = {"price": pred, "model_version": RUN_ID}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
