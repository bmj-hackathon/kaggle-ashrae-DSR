# %%
def preprocess_date_time_cols(df):
    df['date'] = df['timestamp'].dt.date
    logging.info("Added date column".format())
    df["hour"] = df["timestamp"].dt.hour
    logging.info("Added hour column".format())
    # df["day"] = df["timestamp"].dt.day
    df["weekend"] = df["timestamp"].dt.weekday
    logging.info("Added weekend column".format())
    df["month"] = df["timestamp"].dt.month
    logging.info("Added month column".format())
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    logging.info("Added dayofweek column".format())

    return df


# %% Add dates and times
logging.info("Adding basic time features to train and test".format())
train_df = preprocess_date_time_cols(train_df)
test_df = preprocess_date_time_cols(test_df)

#%% Take the ln transform of the targets
train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])
test_df['meter_reading_log1p'] = np.log1p(test_df['meter_reading'])
logging.info("Added meter_reading_log1p [ln(1+x)] column".format())

#%%
if 0:
    # test_df['building_mean'] = test_df['building_id'].map(building_mean)
    test_df['building_median'] = test_df['building_id'].map(building_median)
    # test_df['building_min'] = test_df['building_id'].map(building_min)
    # test_df['building_max'] = test_df['building_id'].map(building_max)
    # test_df['building_std'] = test_df['building_id'].map(building_std)

    print('preprocessing weather...')
    weather_test_df = weather_test_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
    weather_test_df.groupby('site_id').apply(lambda group: group.isna().sum())

    # add_lag_feature(weather_test_df, window=3)
    # add_lag_feature(weather_test_df, window=72)

    print('reduce mem usage...')
    reduce_mem_usage(test_df, use_float16=True)
    reduce_mem_usage(weather_test_df, use_float16=True)

    gc.collect()



    logging.info("Removing building site=0".format())
    building_meta_df[building_meta_df.site_id == 0]
    train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

