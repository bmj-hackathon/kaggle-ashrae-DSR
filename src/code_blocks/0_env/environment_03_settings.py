# Set the environment
SETTINGS = Map()
SETTINGS.data = Map()
SETTINGS.sample = Map()
SETTINGS.features = Map()
SETTINGS.model = Map()

# DATA
SETTINGS.data.path_data_root = Path.cwd() / 'data'
SETTINGS.data.path_data_feather = SETTINGS.data.path_data_root / 'feather'
SETTINGS.data.path_output = Path.cwd() / 'output'
assert SETTINGS.data.path_data_root.exists()
assert SETTINGS.data.path_data_feather.exists()

# FEATURES
SETTINGS.features.psychrometric = True # Add the psychrometric weather features

# SAMPLING
# SETTINGS.sample.drop = 0.9 # Amount of data to drop during dev
SETTINGS.sample.drop = None
# SETTINGS.sample.site = 0 # Set to None/int to execute on only one site
SETTINGS.sample.site = 1 # Set to None/int to sample on only one site
SETTINGS.sample.leakage_subsample = False

# MODEL
SETTINGS.model.folds = 5
SETTINGS.model.num_rounds=1000
SETTINGS.control = Map()
SETTINGS.control.debug = False

logging.info("Settings:".format())
pprint(SETTINGS)