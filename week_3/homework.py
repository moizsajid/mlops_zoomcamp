import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, context
from prefect.task_runners import SequentialTaskRunner
from datetime import datetime
from dateutil.relativedelta import relativedelta
import urllib

@task(retries=3, retry_delay_seconds=60)
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = context.get("logger")
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

def run_model(df, categorical, dv, lr):
    logger = context.get("logger")
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    datetime_object = datetime.strptime(date, "%Y-%m-%d")
    
    train_date = datetime_object - relativedelta(months=2)
    val_date = datetime_object - relativedelta(months=1)

    train_month, train_year = str(train_date.month).zfill(2), str(train_date.year)
    val_month, val_year = str(val_date.month).zfill(2), str(val_date.year)
        
    train_path: str = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{train_year}-{train_month}.parquet"
    val_path: str = f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{val_year}-{val_month}.parquet"

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    run_model(df_val_processed, categorical, dv, lr)

main(date="2021-08-15")
