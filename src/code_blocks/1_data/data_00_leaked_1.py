#%% Load SITE 0
LEAK_SITE = 1
path_leak = SETTINGS.data.path_data_root / 'leakage' / 'site {}'.format(LEAK_SITE)
assert path_leak.exists()

path_leak_feather = path_leak / 'site{}.feather'.format(LEAK_SITE)
assert path_leak_feather.exists()

df_site0_leak = pd.read_feather(path_leak_feather)
logging.info("Loaded site {} leakage: {}".format(LEAK_SITE, path_leak_feather))

#%%
logging.info("Leak summary for site {}:".format(LEAK_SITE))
logging.info("\t{} buildings".format(df_site0_leak['building_id'].nunique()))
leaked_days = (df_site0_leak['timestamp'].max() - df_site0_leak['timestamp'].min()).days / 365
logging.info("\t{:0.1f} years, from {:%Y-%b-%d} to {:%Y-%b-%d}".format(leaked_days, df_site0_leak['timestamp'].min(), df_site0_leak['timestamp'].max()))

#%%
# train_df_leak0_actual.rename({'meter_reading_scraped':'meter_reading'}, axis=1, inplace=True)

# for bldg_id in train_df_leak0_actual['building_id'].unique():
#     print(bldg_id)
#     this_bldg_df = util_data.get_building(train_df_leak0_actual, bldg_id)

#%%

