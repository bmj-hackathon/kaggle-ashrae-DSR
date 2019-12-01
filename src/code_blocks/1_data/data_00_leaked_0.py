#%% Load
path_leak = SETTINGS.data.path_data_root / 'leakage' / 'site 0'
assert path_leak.exists()

path_leak_feather = path_leak / 'site0.feather'
assert path_leak_feather.exists()

df_site0_leak = pd.read_feather(path_leak_feather)
logging.info("Loaded site 0 leakage: {}".format(path_leak_feather))

#%%

logging.info("Site 0 leakage with {} buildings".format(df_site0_leak['building_id'].nunique()))
leaked_days = (df_site0_leak['timestamp'].max() - df_site0_leak['timestamp'].min()).days / 365
logging.info("Site 0 leakage {:0.1f} years, from {:%Y-%b-%d} to {:%Y-%b-%d}".format(leaked_days, df_site0_leak['timestamp'].min(), df_site0_leak['timestamp'].max()))

#%%
train_df_leak0_actual = df_site0_leak.drop('meter_reading_original', axis=1)
train_df_leak0_actual.rename({'meter_reading_scraped':'meter_reading'}, axis=1, inplace=True)

for bldg_id in train_df_leak0_actual['building_id'].unique():
    print(bldg_id)
    this_bldg_df = util_data.get_building(train_df_leak0_actual, bldg_id)

#%%


sample_submission

df_site0_leak

