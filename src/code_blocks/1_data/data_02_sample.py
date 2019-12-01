#%%
if SETTINGS.sample.drop:
    drop_after = int((1-SETTINGS.data.drop) * len(train_df))
    train_df = train_df.iloc[0:drop_after]
    logging.info("Dropped data, train_df reduced to {}".format(train_df.shape))

#%%
if SETTINGS.sample.site != None:
    logging.info("Down sampling to site {}".format(SETTINGS.sample.site))
    train_df = util_data.select_buildings_on_site(train_df, building_meta_df, SETTINGS.sample.site)

#%%
if SETTINGS.sample.leakage_subsample:
    pass
#%%

#%%
def get_mem(df):
    return df.memory_usage().sum() / 1024 ** 2

# Train data
train_merge = None
logging.info("Original train_df {:0.1f} MB".format(get_mem(train_df)))
train_merge = train_df.merge(building_meta_df, on='building_id', how='left')
logging.info("Merged meta to train_merge {:0.1f} MB".format(get_mem(train_merge)))

train_merge = train_merge.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
logging.info("Merged weather to train_merge {:0.1f} MB".format(get_mem(train_merge)))

# Test data
test_merge = None
logging.info("Original test_df {:0.1f} MB".format(get_mem(test_df)))
test_merge = test_df.merge(building_meta_df, on='building_id', how='left')
logging.info("Merged meta to test_merge {:0.1f} MB".format(get_mem(test_merge)))

test_merge = test_merge.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
logging.info("Merged weather to test_merge {:0.1f} MB".format(get_mem(test_merge)))

r = train_merge.head()
r = test_merge.head()
