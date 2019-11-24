# %%
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype



def get_building(df_meter_records, bldg_id):
    """
    Given the flat records for each meter-building-site

    Break out a summary energy profile DF for each building with the columns;
    Timestamp - Meter 1 - Meter 1 - Meter 1 - Meter 1
    Renames meter columns to proper energy sources
    """

    # Select the building records
    bldg_df = df_meter_records[df_meter_records.loc[:, 'building_id'] == bldg_id]
    bldg_df.drop('building_id', axis=1, inplace=True)
    logging.info("Building {} with {} records over {} meters".format(bldg_id, len(bldg_df), bldg_df['meter'].nunique()))

    # Pivot the records to meter-meter-meter-meter format
    bldg_df_pivot = bldg_df.pivot(index='timestamp', columns='meter', values='meter_reading')

    # For convenience, create empty columns for missing meters
    for meter_col in range(4):
        if meter_col not in bldg_df_pivot.columns:
            bldg_df_pivot[meter_col] = int(0)

    # bldg_df_pivot.memory_usage().sum() / 1024 ** 2 # Size in MB
    bldg_df_pivot.rename(columns={0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}, inplace=True)
    logging.info("{} to {}".format(bldg_df_pivot.index[0], bldg_df_pivot.index[-1]))
    assert bldg_df_pivot.index.is_monotonic_increasing

    return bldg_df_pivot

# for i in range(1000):
# bldg_id = i
# this_bldg = get_building(train_df, bldg_id)


#%%
def get_site_weather(weather_df, site_id):

    # Select the building records
    site_weather_df = weather_df[weather_df.loc[:, 'site_id'] == site_id]
    site_weather_df.drop('site_id', axis=1, inplace=True)
    logging.info("Site {} with {} records".format(site_id, len(site_weather_df)))
    site_weather_df.set_index('timestamp', drop=True, inplace=True)
    logging.info("{} to {}".format(site_weather_df.index[0], site_weather_df.index[-1]))
    assert site_weather_df.index.is_monotonic_increasing

    return site_weather_df

# site_id = 0
# for i in range(16):
#     r = get_site_weather(weather_train_df,i)
# r = weather_train_df.head()

#%%
def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    logging.debug('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
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
    logging.debug('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logging.info('Memory decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
#%%
class Map(dict):
    """
    Example:
    mj = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]
