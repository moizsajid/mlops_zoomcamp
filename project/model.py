import os
import pickle

import mlflow
import pandas as pd

RUN_ID = os.getenv("RUN_ID", "f0033d09b079439890630489db4b3f5c")


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def load_model():
    logged_model = f"s3://mlflow-models-mlopszoomcamp/1/{RUN_ID}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


class ModelService:
    def __init__(self, model):
        self.model = model

    def prepare_features(self, car):
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

    def predict(self, car):
        features = self.prepare_features(car)
        pred = self.model.predict(features)
        return float(pred)


def init():
    model = load_model()

    model_service = ModelService(model=model)

    return model_service
