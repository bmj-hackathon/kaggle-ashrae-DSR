
# %%
logging.info("Adding building_median log1p feature".format())

df_group = df_train.groupby('building_id')['meter_reading_log1p']
#building_mean = df_group.mean().astype(np.float16)
building_median = df_group.median().astype(np.float16)
#building_min = df_group.min().astype(np.float16)
#building_max = df_group.max().astype(np.float16)
#building_std = df_group.std().astype(np.float16)

#train_df['building_mean'] = train_df['building_id'].map(building_mean)
df_train['building_median'] = train_df['building_id'].map(building_median)
#train_df['building_min'] = train_df['building_id'].map(building_min)
#train_df['building_max'] = train_df['building_id'].map(building_max)
#train_df['building_std'] = train_df['building_id'].map(building_std)
del df_group
logging.info("Added building_median log1p feature".format())

# %%
#building_mean.head()

# %% [markdown]
# # Fill Nan value in weather dataframe by interpolation
#
#
# weather data has a lot of NaNs!!
#
# ![](http://)I tried to fill these values by **interpolating** data.

# %%
# weather_train_df.head()

# %%
# weather_train_df.describe()

# %%
# weather_train_df.isna().sum()

# %%
# weather_train_df.shape

# %%
logging.info("Interpolating over NaNs in weather_train_df".format())
# weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())
weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))

# %%
# weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())


# %% [markdown]
# Seems number of nan has reduced by `interpolate` but some property has never appear in specific `site_id`, and nan remains for these features.

# %% [markdown]
# ## lags
#
# Adding some lag feature

# %%
def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_std_lag{window}'] = lag_std[col]


# %%
# skip lag feature to save memory
#add_lag_feature(weather_train_df, window=3)
#add_lag_feature(weather_train_df, window=72)

# %%
weather_train_df.head()

# %%
# categorize primary_use column to reduce memory on merge...
logging.info("Convert primary_use to Categorical type".format())
primary_use_list = building_meta_df['primary_use'].unique()
primary_use_dict = {key: value for value, key in enumerate(primary_use_list)}
print('primary_use_dict: ', primary_use_dict)
building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)
gc.collect()

# %%
reduce_mem_usage(train_df, use_float16=True)
reduce_mem_usage(building_meta_df, use_float16=True)
reduce_mem_usage(weather_train_df, use_float16=True)

# %%
building_meta_df.head()
