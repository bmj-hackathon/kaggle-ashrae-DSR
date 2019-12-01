#%% Load SITE 0

leak_dfs = dict()

LEAK_SITE = 0
path_leak = SETTINGS.data.path_data_root / 'leakage' / 'site {}'.format(LEAK_SITE)
assert path_leak.exists()

path_leak_feather = path_leak / 'site{}.feather'.format(LEAK_SITE)
assert path_leak_feather.exists()

leak_dfs[LEAK_SITE] = pd.read_feather(path_leak_feather)
logging.info("Loaded site {} leakage: {}".format(LEAK_SITE, path_leak_feather))

#%%
logging.info("Leak summary for site {}:".format(LEAK_SITE))
logging.info("\t{} buildings".format(leak_dfs[LEAK_SITE]['building_id'].nunique()))
leaked_years = (leak_dfs[LEAK_SITE]['timestamp'].max() - leak_dfs[LEAK_SITE]['timestamp'].min()).days / 365
logging.info("\t{:0.1f} years, from {:%Y-%b-%d} to {:%Y-%b-%d}".format(leaked_days, leak_dfs[LEAK_SITE]['timestamp'].min(), leak_dfs[LEAK_SITE]['timestamp'].max()))



#%%
# train_df_leak0_actual.rename({'meter_reading_scraped':'meter_reading'}, axis=1, inplace=True)

# for bldg_id in train_df_leak0_actual['building_id'].unique():
#     print(bldg_id)
#     this_bldg_df = util_data.get_building(train_df_leak0_actual, bldg_id)

#%%

