#%% Train data
logging.info(" *** Step 2: Load data *** ".format())

# Train
train_df = pd.read_feather(SETTINGS.data.path_data_root / 'train.feather')
logging.info("Loaded: train_df {} with {} buildings, {:0.1f} MB".format(train_df.shape, train_df.loc[:, 'building_id'].nunique(), train_df.memory_usage().sum() / 1024 ** 2))
r1 = train_df.head()
train_df = reduce_mem_usage(train_df)

# Test
test_df = pd.read_feather(SETTINGS.data.path_data_root / 'test.feather')
logging.info("Loaded: test_df {} with {} buildings, {:0.1f} MB".format(test_df.shape, test_df.loc[:, 'building_id'].nunique(), test_df.memory_usage().sum() / 1024 ** 2))
r2 = test_df.head()
test_df = reduce_mem_usage(test_df)

# Weather train
weather_train_df = pd.read_feather(SETTINGS.data.path_data_root/'weather_train.feather')
logging.info("Loaded: weather_train_df {}".format(weather_train_df.shape))
r = weather_train_df.head()
weather_train_df = reduce_mem_usage(weather_train_df)

# Weather test
weather_test_df = pd.read_feather(SETTINGS.data.path_data_root/'weather_test.feather')
logging.info("Loaded: weather_test_df {}".format(weather_test_df.shape))
weather_test_df = reduce_mem_usage(weather_test_df)

# Meta
building_meta_df = pd.read_feather(SETTINGS.data.path_data_root/'building_metadata.feather')
logging.info("Loaded: building_meta_df {}".format(building_meta_df.shape))
r = building_meta_df.head()
building_meta_df.set_index('building_id',inplace=True, drop=True)
building_meta_df = reduce_mem_usage(building_meta_df)

# Sample
sample_submission = pd.read_feather(os.path.join(SETTINGS.data.path_data_root, 'sample_submission.feather'))
logging.info("Loaded: sample_submission {}".format(sample_submission.shape))
sample_submission = reduce_mem_usage(sample_submission)

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