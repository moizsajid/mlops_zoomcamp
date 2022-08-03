import sys

import pandas as pd
from datetime import datetime
from pandas.testing import assert_frame_equal

import batch


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),     
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    data_frame = pd.DataFrame(data, columns=columns)
    categorical = ['PUlocationID', 'DOlocationID']

    actual = batch.prepare_data(data_frame, categorical)

    expected_data = [
        (-1, -1, dt(1, 2), dt(1, 10), 8.0),
        (1, 1, dt(1, 2), dt(1, 10), 8.0),
    ]

    expected = pd.DataFrame(expected_data, columns=['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime', 'duration'])
    expected[categorical] = expected[categorical].astype('str')

    assert_frame_equal(actual, expected)
