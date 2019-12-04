## Memory Optimization
# https://www.kaggle.com/rohanrao/ashrae-divide-and-conquer @vopani
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

import gc
import numpy as np
import pandas as pd

from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pathlib import Path
import zipfile
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


DATA_PATH = Path.cwd() / 'data/raw'
assert DATA_PATH.exists(), DATA_PATH

DATA_FEATHER_PATH = Path.cwd() / 'data/feather'
DATA_FEATHER_PATH.mkdir(parents=True, exist_ok=True)


ZIPPED = False
MERGE = True
MERGE_ON_NAN_CLEANING = True

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
    train_df = pd.read_csv(DATA_PATH / 'train.csv.zip')
    test_df = pd.read_csv(DATA_PATH / 'test.csv.zip')
    weather_train_df = pd.read_csv(DATA_PATH / 'weather_train.csv.zip')
    weather_test_df = pd.read_csv(DATA_PATH / 'weather_test.csv.zip')
    building_meta_df = pd.read_csv(DATA_PATH / 'building_metadata.csv.zip')
    sample_submission = pd.read_csv(DATA_PATH / 'sample_submission.csv.zip')

#%%
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

if MERGE and not MERGE_ON_NAN_CLEANING:
    train_merged = train_df.merge(building_meta_df, how='left', on='building_id')
    test_merged = test_df.merge(building_meta_df, how='left', on='building_id')

    train_merged = train_merged.merge(weather_train_df, how='left', on=['site_id', 'timestamp'])
    test_merged = test_merged.merge(weather_test_df, how='left', on=['site_id', 'timestamp'])

    train_merged.to_feather(DATA_FEATHER_PATH /'train_merged.feather')
    test_merged.to_feather(DATA_FEATHER_PATH /'test_merged.feather')

def weather_fillna(weather_df):
    weather_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature',
                    'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']

    weather_df[weather_cols] = weather_df.groupby(['site_id'])[weather_cols].transform(
        lambda x: x.interpolate(method='spline',
                                limit_direction='both',
                                order=2))

    iter_impute = IterativeImputer(max_iter=100, random_state=42)

    weather_df1 = weather_df.drop('timestamp', axis=1)
    weather_df_imputed = iter_impute.fit_transform(weather_df1)

    weather_df_imputed = pd.DataFrame(weather_df_imputed, columns=weather_df1.columns)

    weather_df.loc[:, weather_df1.columns] = weather_df_imputed.loc[:, weather_df1.columns].values

    del weather_df1, weather_df_imputed
    return weather_df


if MERGE_ON_NAN_CLEANING:
    weather_train_df = weather_fillna(weather_train_df)
    weather_test_df = weather_fillna(weather_test_df)

    train_merged = train_df.merge(building_meta_df, how='left', on='building_id')
    test_merged = test_df.merge(building_meta_df, how='left', on='building_id')

    train_merged = train_merged.merge(weather_train_df, how='left', on=['site_id', 'timestamp'])
    test_merged = test_merged.merge(weather_test_df, how='left', on=['site_id', 'timestamp'])

    train_merged.to_feather(DATA_FEATHER_PATH / 'train_merged.feather')
    test_merged.to_feather(DATA_FEATHER_PATH / 'test_merged.feather')

