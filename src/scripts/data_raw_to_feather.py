## Memory Optimization
# https://www.kaggle.com/rohanrao/ashrae-divide-and-conquer @vopani
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

import gc
import numpy as np
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pathlib import Path
import zipfile

DATA_PATH = 'data/raw'
DATA_PATH = Path(DATA_PATH)
DATA_PATH = DATA_PATH.expanduser()
assert DATA_PATH.exists(), DATA_PATH

DATA_FEATHER_PATH ='data/feather'
DATA_FEATHER_PATH = Path(DATA_FEATHER_PATH)
DATA_FEATHER_PATH = DATA_FEATHER_PATH.expanduser()
DATA_FEATHER_PATH.mkdir(parents=True, exist_ok=True)


ZIPPED = False
MERGE = True

def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]):
            # skip datetime type
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df
#%%

# %%
from pathlib import Path
import zipfile
DATA_PATH = '~/ashrae/data/raw'
DATA_PATH = Path(DATA_PATH)
DATA_PATH = DATA_PATH.expanduser()
assert DATA_PATH.exists(), DATA_PATH

DATA_FEATHER_PATH ='~/ashrae/data/feather'
DATA_FEATHER_PATH = Path(DATA_FEATHER_PATH)
DATA_FEATHER_PATH = DATA_FEATHER_PATH.expanduser()
DATA_FEATHER_PATH.mkdir(exist_ok=True)
assert DATA_FEATHER_PATH.exists()

# zipfile.ZipFile(DATA_PATH).infolist()

#%%
ZIPPED = False

# %%time
if ZIPPED:
    with zipfile.ZipFile(DATA_PATH) as zf:
        with zf.open('train.csv') as zcsv:
            train_df = pd.read_csv(zcsv)
        with zf.open('test.csv') as zcsv:
            test_df = pd.read_csv(zcsv)
        with zf.open('weather_train.csv') as zcsv:
            weather_train_df = pd.read_csv(zcsv)
        with zf.open('weather_test.csv') as zcsv:
            weather_test_df = pd.read_csv(zcsv)
        with zf.open('building_metadata.csv') as zcsv:
            building_meta_df = pd.read_csv(zcsv)
        with zf.open('sample_submission.csv') as zcsv:
            sample_submission = pd.read_csv(zcsv)
else:
    train_df = pd.read_csv(DATA_PATH / 'train.csv')
    test_df = pd.read_csv(DATA_PATH / 'test.csv')
    weather_train_df = pd.read_csv(DATA_PATH / 'weather_train.csv')
    weather_test_df = pd.read_csv(DATA_PATH / 'weather_test.csv')
    building_meta_df = pd.read_csv(DATA_PATH / 'building_metadata.csv')
    sample_submission = pd.read_csv(DATA_PATH / 'sample_submission.csv')


train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])
weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])

gc.collect()


reduce_mem_usage(train_df)
reduce_mem_usage(test_df)
reduce_mem_usage(building_meta_df)
reduce_mem_usage(weather_train_df)
reduce_mem_usage(weather_test_df)

train_df.to_feather(DATA_FEATHER_PATH /'train.feather')
test_df.to_feather(DATA_FEATHER_PATH / 'test.feather')
weather_train_df.to_feather(DATA_FEATHER_PATH /'weather_train.feather')
weather_test_df.to_feather(DATA_FEATHER_PATH /'weather_test.feather')
building_meta_df.to_feather(DATA_FEATHER_PATH /'building_metadata.feather')
sample_submission.to_feather(DATA_FEATHER_PATH /'sample_submission.feather')

if MERGE:
    train_merged = train_df.merge(building_meta_df, how='left', on='building_id')
    test_merged = test_df.merge(building_meta_df, how='left', on='building_id')

    train_merged = train_merged.merge(weather_train_df, how='left', on=['site_id', 'timestamp'])
    test_merged = test_merged.merge(weather_test_df, how='left', on=['site_id', 'timestamp'])

    train_merged.to_feather(DATA_FEATHER_PATH /'train_merged.feather')
    test_merged.to_feather(DATA_FEATHER_PATH /'test_merged.feather')
