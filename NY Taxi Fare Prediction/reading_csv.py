import pandas as pd
import dask.dataframe as dd
import os
import subprocess

# with open('train.csv') as file:
#     n_rows = len(file.readlines())

# print(f'Exact number of rows: {n_rows}')

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])+1

df_tmp = pd.read_csv('train.csv', nrows=5)

print(df_tmp.info())

traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str',
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())

chunksize = 5_000_000

df_list = []

for df_chunk in pd.read_csv('train.csv', usecols=cols, dtype=traintypes, chunksize=chunksize):
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    # str.slice in this example gets the first 16 characters of the string
    # this is consistent with using a datatype of float32
    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    df_list.append(df_chunk)