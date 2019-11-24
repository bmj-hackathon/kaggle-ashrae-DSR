
# %%

category_cols = ['building_id', 'site_id', 'primary_use']  # , 'meter'
feature_cols = ['square_feet', 'year_built'] + [
    'hour', 'weekend',  # 'month' , 'dayofweek'
    'building_median'] + [
                   'air_temperature', 'cloud_coverage',
                   'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
                   'wind_direction', 'wind_speed', ]


#     'air_temperature_mean_lag72',
#     'air_temperature_max_lag72', 'air_temperature_min_lag72',
#     'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',
#     'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',
#     'sea_level_pressure_mean_lag72', 'wind_direction_mean_lag72',
#     'wind_speed_mean_lag72', 'air_temperature_mean_lag3',
#     'air_temperature_max_lag3',
#     'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',
#     'dew_temperature_mean_lag3',
#     'precip_depth_1_hr_mean_lag3', 'sea_level_pressure_mean_lag3',
#     'wind_direction_mean_lag3', 'wind_speed_mean_lag3']
# %% [markdown]
# # Train model
#
# To win in kaggle competition, how to evaluate your model is important.
# What kind of cross validation strategy is suitable for this competition? This is time series data, so it is better to consider time-splitting.
#
# However this notebook is for simple tutorial, so I will proceed with KFold splitting without shuffling, so that at least near-term data is not included in validation.

# %%
def create_X_y(train_df, target_meter):
    target_train_df = train_df[train_df['meter'] == target_meter]
    target_train_df = target_train_df.merge(building_meta_df, on='building_id', how='left')
    target_train_df = target_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
    X_train = target_train_df[feature_cols + category_cols]
    y_train = target_train_df['meter_reading_log1p'].values

    del target_train_df
    return X_train, y_train

# %%
# folds = 5
seed = 666
shuffle = False
kf = sk.model_selection.KFold(n_splits=SETTINGS.model.folds, shuffle=shuffle, random_state=seed)
oof_total = 0

# %% [markdown]
# # Train model by each meter type

# %%
target_meter = 0
X_train, y_train = create_X_y(train_df, target_meter=target_meter)
y_valid_pred_total = np.zeros(X_train.shape[0])
gc.collect()
print('target_meter', target_meter, X_train.shape)

cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]
print('cat_features', cat_features)
