#%% Load
path_leak = SETTINGS.data.path_data_root / 'leakage' / 'site 0'
assert path_leak.exists()

path_leak_csv = path_leak / 'site0.csv.gz'
path_leak_feather = path_leak / 'site0.feather'
if path_leak_feather.exists():
    train_df_leak0 = pd.read_feather(path_leak_feather)
    logging.info("Loaded site 0 leakage: {}".format(path_leak_feather))
else: # Create the feather file
    assert path_leak_csv.exists()
    train_df_leak0 = pd.read_csv(path_leak_csv)

    # Reduce
    train_df_leak0['timestamp'] = pd.to_datetime(train_df_leak0['timestamp'])
    reduce_mem_usage(train_df_leak0)
    # train_df.head()
    # train_df_leak0.info()
    # train_df.info()

    # Save
    train_df_leak0.to_feather(path_leak_feather)
    logging.info("Saved {}".format(path_leak_feather))

#%%

