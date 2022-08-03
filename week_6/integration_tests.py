import pandas as pd
from datetime import datetime


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)

categorical = ['PUlocationID', 'DOlocationID']

expected_data = [
    (-1, -1, dt(1, 2), dt(1, 10), 8.0),
    (1, 1, dt(1, 2), dt(1, 10), 8.0),
]

expected = pd.DataFrame(expected_data, columns=['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime', 'duration'])
expected[categorical] = expected[categorical].astype('str')

options = {
    'client_kwargs': {
        'endpoint_url': "http://localhost:4566"
    }
}

year = 2021
month = 1

input_file = f"s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"

expected.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)
