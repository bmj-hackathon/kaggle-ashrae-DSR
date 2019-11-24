# Set the environment
SETTINGS = Map()
SETTINGS.data = Map()
SETTINGS.model = Map()

SETTINGS.data.path_data_root = Path.cwd() / 'data'
SETTINGS.data.use_ucf = True
SETTINGS.data.path_output = Path.cwd() / 'output'

SETTINGS.model.folds = 5
SETTINGS.model.num_rounds=1000
SETTINGS.control = Map()
SETTINGS.control.debug = False


logging.info("Settings:".format())
print(SETTINGS)