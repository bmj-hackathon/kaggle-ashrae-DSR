#%% Train data
logging.info(" *** Step 2: Load data *** ".format())

# Train #############
train_df = pd.read_feather(SETTINGS.data.path_data_feather / 'train.feather')
logging.info("Loaded: train_df {} with {} buildings, {:0.1f} MB".format(train_df.shape, train_df.loc[:, 'building_id'].nunique(), train_df.memory_usage().sum() / 1024 ** 2))

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

# Sample submission
df_submission = pd.read_feather(os.path.join(SETTINGS.data.path_data_feather, 'sample_submission.feather'))
logging.info("Loaded: sample_submission {}".format(df_submission.shape))
sample_submission = reduce_mem_usage(df_submission)

if SETTINGS.features.psychrometric:
    logging.info("Feature: Psychrometric".format())
    # Weather test Psychro #############
    weather_test_psychro_df = pd.read_feather(SETTINGS.data.path_data_root / 'feature_psychrometric' / 'weather_test_psychrometric.feather')
    logging.info("Loaded: weather_test_psychro_df {}".format(weather_test_psychro_df.shape))

    weather_train_psychro_df = pd.read_feather(SETTINGS.data.path_data_root / 'feature_psychrometric' / 'weather_train_psychrometric.feather')
    logging.info("Loaded: weather_train_psychro_df {}".format(weather_train_psychro_df.shape))
    # weather_test_psychro_df.index
    # weather_test_df.index
    # weather_test_df.merge(weather_test_psychro_df, )
