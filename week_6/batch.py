#!/usr/bin/env python
# coding: utf-8

"""
File for running model prediction on processed data
"""

import os
import sys
import pickle
import pandas as pd


def get_input_path(year, month):
    default_input_pattern = f'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/nyc-tlc/fhv/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = f's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename, options):
    """
    Read data from file
    """
    if options["client_kwargs"]["endpoint_url"]:
        data_frame = pd.read_parquet(filename, storage_options=options)
    else:
        data_frame = pd.read_parquet(filename)

    return data_frame


def prepare_data(data_frame, categorical):
    """
    Process data
    """
    data_frame['duration'] = data_frame.dropOff_datetime - data_frame.pickup_datetime
    data_frame['duration'] = data_frame.duration.dt.total_seconds() / 60

    data_frame = data_frame[(data_frame.duration >= 1) & (data_frame.duration <= 60)].copy()

    data_frame[categorical] = data_frame[categorical].fillna(-1).astype('int').astype('str')

    return data_frame


def save_data(data_frame, output_file, options):

    data_frame.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )


def main(year, month):
    """
    main function
    """
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    options = {
        'client_kwargs': {
            'endpoint_url': "http://localhost:4566"
        }
    }

    with open('model.bin', 'rb') as f_in:
        dict_vectorizer, logistic_regression = pickle.load(f_in)

    categorical = ['PUlocationID', 'DOlocationID']

    data_frame = read_data(input_file, options)
    data_frame['ride_id'] = f'{year:04d}/{month:02d}_' + data_frame.index.astype('str')

    dicts = data_frame[categorical].to_dict(orient='records')
    x_val = dict_vectorizer.transform(dicts)
    y_pred = logistic_regression.predict(x_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = data_frame['ride_id']
    df_result['predicted_duration'] = y_pred

    #df_result.to_parquet(output_file, engine='pyarrow', index=False)

    save_data(df_result, output_file, options)

if __name__=="__main__":
    year_arg = int(sys.argv[1])
    month_arg = int(sys.argv[2])
    main(year_arg, month_arg)
