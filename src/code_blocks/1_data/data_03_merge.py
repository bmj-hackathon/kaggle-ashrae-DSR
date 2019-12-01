#%%
def get_mem(df):
    return df.memory_usage().sum() / 1024 ** 2

# Train data
df_train_merged = None
logging.info("Original train_df {:0.1f} MB".format(get_mem(train_df)))
df_train_merged = train_df.merge(building_meta_df, on='building_id', how='left')
logging.info("Merged meta to train_merge {:0.1f} MB".format(get_mem(df_train_merged)))

df_train_merged = df_train_merged.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
logging.info("Merged weather to train_merge {:0.1f} MB".format(get_mem(df_train_merged)))

# Test data
df_test_merged = None
logging.info("Original test_df {:0.1f} MB".format(get_mem(test_df)))
df_test_merged = test_df.merge(building_meta_df, on='building_id', how='left')
logging.info("Merged meta to test_merge {:0.1f} MB".format(get_mem(df_test_merged)))

df_test_merged = df_test_merged.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
logging.info("Merged weather to test_merge {:0.1f} MB".format(get_mem(df_test_merged)))

r = df_train_merged.head()
r = df_test_merged.head()

remove_vars = list()
remove_vars += ['train_df', 'weather_train_df', 'test_df', 'weather_test_df']
remove_vars += ['weather_test_psychro_df', 'weather_train_psychro_df']
for var in remove_vars:
    logging.info("Removing {}, {:0.1f} MB".format(var, sys.getsizeof(globals()[var]) / 1012 ** 2 ))
    if var in globals():
        del globals()[var]
