#%% Load
from pathlib import Path
import pandas as pd
from src.utils.utility_classes import reduce_mem_usage
import logging

path_leak = Path.cwd() / 'data' / 'leakage' / 'site 0'
assert path_leak.exists()

path_leak_data = path_leak / 'site0.csv.gz'
path_leak_feather = path_leak / 'site0.feather'

assert path_leak_data.exists()
df_leak = pd.read_csv(path_leak_data)

# Reduce
df_leak['timestamp'] = pd.to_datetime(df_leak['timestamp'])
df_leak = reduce_mem_usage(df_leak)
df_leak.drop('meter_reading_original', axis=1, inplace=True)

# Save
df_leak.to_feather(path_leak_feather)
logging.info("Saved {}".format(path_leak_feather))
print("Saved {}".format(path_leak_feather))
