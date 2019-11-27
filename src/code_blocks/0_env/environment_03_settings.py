# Set the environment
SETTINGS = Map()
SETTINGS.data = Map()
SETTINGS.model = Map()

# DATA
SETTINGS.data.path_data_root = Path.cwd() / 'data' / 'feather'
SETTINGS.data.path_output = Path.cwd() / 'output'
assert SETTINGS.data.path_data_root.exists()
SETTINGS.data.drop = 0.9 # Amount of data to drop during dev

# MODEL
SETTINGS.model.folds = 5
SETTINGS.model.num_rounds=1000
SETTINGS.control = Map()
SETTINGS.control.debug = False

#
SETTINGS.data.path_data_root = Path.cwd() / 'data' / 'feather'
SETTINGS.data.use_ucf = True
SETTINGS.data.path_output = Path.cwd() / 'output'

logging.info("Settings:".format())
pprint(SETTINGS)