#%% Train data
logging.info(" *** Step 2: Load data *** ".format())

# Train #############
train_df = pd.read_feather(SETTINGS.data.path_data_feather / 'train.feather')
logging.info("Loaded: train_df {} with {} buildings, {:0.1f} MB".format(train_df.shape, train_df.loc[:, 'building_id'].nunique(), train_df.memory_usage().sum() / 1024 ** 2))
if SETTINGS.data.drop:
    drop_after = int((1-SETTINGS.data.drop) * len(train_df))
    train_df = train_df.iloc[0:drop_after]
    logging.info("Dropped data, train_df reduced to {}".format(train_df.shape))

# Test #############
test_df = pd.read_feather(SETTINGS.data.path_data_feather / 'test.feather')
logging.info("Loaded: test_df {} with {} buildings, {:0.1f} MB".format(test_df.shape, test_df.loc[:, 'building_id'].nunique(), test_df.memory_usage().sum() / 1024 ** 2))

# Weather train #############
weather_train_df = pd.read_feather(SETTINGS.data.path_data_feather/'weather_train.feather')
logging.info("Loaded: weather_train_df {}".format(weather_train_df.shape))

# Weather test #############
weather_test_df = pd.read_feather(SETTINGS.data.path_data_feather/'weather_test.feather')
logging.info("Loaded: weather_test_df {}".format(weather_test_df.shape))

# Meta #############
building_meta_df = pd.read_feather(SETTINGS.data.path_data_feather/'building_metadata.feather')
logging.info("Loaded: building_meta_df {}".format(building_meta_df.shape))
building_meta_df.set_index('building_id', inplace=True, drop=True)

# Sample
# sample_submission = pd.read_feather(os.path.join(SETTINGS.data.path_data_root, 'sample_submission.feather'))
# logging.info("Loaded: sample_submission {}".format(sample_submission.shape))
# sample_submission = reduce_mem_usage(sample_submission)
#%%
if SETTINGS.data.site:
    util_data.select_buildings_on_site(train_df, building_meta_df, SETTINGS.data.site)

#%%
# train_merge = train_df_data.merge(building_meta_df, on='building_id', how='left')
# train_merge.memory_usage().sum() / 1024 ** 2
# test_merge = test_df_data.merge(building_meta_df, on='building_id', how='left')
# test_merge.memory_usage().sum() / 1024 ** 2
#
# train_df = train_merge.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
# train_df.memory_usage().sum() / 1024 ** 2
# r = train_df.head()
#
# test_df = test_merge.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
# test_df.memory_usage().sum() / 1024 ** 2
# r = test_df.head()

# train2.info()

# del