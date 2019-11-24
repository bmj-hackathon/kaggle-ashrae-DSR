

#%%
import os
from pathlib import Path

#%% ===========================================================================
# Standard imports
# =============================================================================
import os
from pathlib import Path
import sys
import zipfile
import gc
import time
from pprint import pprint
from functools import reduce
from collections import defaultdict
import json
import yaml
import inspect
import gc
import random

#%% ===========================================================================
# Basic imports
# =============================================================================
import tqdm

#%% ===========================================================================
# ML imports
# =============================================================================
import numpy as np
print('numpy {} as np'.format(np.__version__))
import pandas as pd
print('pandas {} as pd'.format(pd.__version__))
from sklearn_pandas import DataFrameMapper
import sklearn as sk
print('sklearn {} as sk'.format(sk.__version__))

import sklearn.preprocessing
import sklearn.metrics
import sklearn.linear_model
import sklearn.pipeline
import sklearn.model_selection
import sklearn.ensemble
import sklearn.feature_extraction
import sklearn.decomposition
import sklearn.compose
import sklearn.utils

# Models
import lightgbm as lgb
print("lightgbm", lgb.__version__)
import xgboost as xgb
print("xgboost", xgb.__version__)
# from catboost import CatBoostClassifier
import catboost as catb
print("catboost", catb.__version__)

#%% ===========================================================================
# Plotting
# =============================================================================
import matplotlib as mpl
print('matplotlib {} as mpl'.format(mpl.__version__))
import matplotlib.pyplot as plt
print('matplotlib.pyplot as plt'.format())
import seaborn as sns
print('seaboarn {} as sns'.format(sns.__version__))

# from IPython.core.display import display, HTML

# --- plotly ---
import plotly.io as pio
pio.renderers.default = "browser"
# from plotly import tools, subplots
import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff


