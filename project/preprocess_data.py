import argparse
import os
import pickle

import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.model_selection import train_test_split


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(raw_data_path):
    df = pd.read_csv(raw_data_path)
    df = df.drop_duplicates(keep=False)
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

    X = df.copy()
    X.drop(columns=["Price", "ID"], inplace=True)
    y = df["Price"]

    return X, y


def preprocess(X: np.ndarray, be: BinaryEncoder, fit_be: bool = False):

    yrs = list(X["Prod. year"].sort_values().unique())
    mapuni = {j: i for i, j in enumerate(yrs)}

    if fit_be:
        X = be.fit_transform(X)
    else:
        X = be.transform(X)

    X["Prod. year"] = X["Prod. year"].replace(mapuni)
    X["Doors"] = X["Doors"].replace({"2-3": 0, "4-5": 1, ">5": 2})
    X["Leather interior"] = X["Leather interior"].replace({"Yes": 1, "No": 0})
    X["Wheel"] = X["Wheel"].replace({"Left wheel": 1, "Right-hand drive": 0})

    X["Cylinders"] = X["Cylinders"].astype("i")
    X["Airbags"] = X["Airbags"].astype("i")

    return X, be


def run(raw_data_path: str, dest_path: str):

    X, y = read_dataframe(raw_data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0, test_size=0.1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, random_state=0, test_size=0.1
    )

    be = BinaryEncoder(
        cols=[
            "Manufacturer",
            "Model",
            "Category",
            "Fuel type",
            "Gear box type",
            "Drive wheels",
            "Color",
        ],
        drop_invariant=False,
        return_df=True,
    )

    X_train, be = preprocess(X_train, be, True)
    X_val, be = preprocess(X_val, be, False)
    X_test, be = preprocess(X_test, be, False)

    # create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # save dictvectorizer and datasets
    dump_pickle(be, os.path.join(dest_path, "be.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        help="the location where the raw car price prediction data was saved",
        default="data/car_price_prediction.csv",
    )
    parser.add_argument(
        "--dest_path",
        help="the location where the resulting files will be saved.",
        default="./output",
    )
    args = parser.parse_args()

    run(args.raw_data_path, args.dest_path)
