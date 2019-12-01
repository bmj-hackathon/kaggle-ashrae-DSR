#%%
if SETTINGS.sample.drop:
    drop_after = int((1-SETTINGS.data.drop) * len(train_df))
    train_df = train_df.iloc[0:drop_after]
    logging.info("Dropped data, train_df reduced to {}".format(train_df.shape))

#%%
if SETTINGS.sample.site != None:
    logging.info("Down sampling to site {}".format(SETTINGS.sample.site))
    train_df = util_data.select_buildings_on_site(train_df, building_meta_df, SETTINGS.sample.site)
    test_df = util_data.select_buildings_on_site(test_df, building_meta_df, SETTINGS.sample.site)

#%%
if SETTINGS.sample.leakage_subsample:
    pass
