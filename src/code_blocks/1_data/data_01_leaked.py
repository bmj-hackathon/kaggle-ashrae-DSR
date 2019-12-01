#%%

def load_leaked_data(LEAK_SITE):
    path_leak = SETTINGS.data.path_data_root / 'leakage' / 'site {}'.format(LEAK_SITE)
    assert path_leak.exists()

    path_leak_feather = path_leak / 'site{}.feather'.format(LEAK_SITE)
    assert path_leak_feather.exists()

    leak_df= pd.read_feather(path_leak_feather)
    logging.info("Loaded site {} leakage: {}".format(LEAK_SITE, path_leak_feather))

    logging.info("Leak summary for site {}:".format(LEAK_SITE))
    logging.info("\t{} records".format(len(leak_df['building_id'])))
    logging.info("\t{} buildings".format(leak_df['building_id'].nunique()))
    leaked_years = (leak_df['timestamp'].max() - leak_df['timestamp'].min()).days / 365
    logging.info("\t{:0.1f} years, from {:%Y-%b-%d} to {:%Y-%b-%d}".format(
        leaked_years, leak_df['timestamp'].min(), leak_df['timestamp'].max()))
    return leak_df

#%%
leak_dfs = dict()

leak_sites = [0, 1, 2, 4, 15]
for l_site in leak_sites:
    leak_dfs[l_site] = load_leaked_data(l_site)
logging.info("Loaded leaked data from sites {}".format(leak_sites))

#%%
# train_df_leak0_actual.rename({'meter_reading_scraped':'meter_reading'}, axis=1, inplace=True)

# for bldg_id in train_df_leak0_actual['building_id'].unique():
#     print(bldg_id)
#     this_bldg_df = util_data.get_building(train_df_leak0_actual, bldg_id)

#%%

