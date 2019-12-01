# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.insert(0, '../../')

# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import build_pipeline
from src.utils import ashrae_transformers

# %reload_ext autoreload

# %%
# !ls ../../

# %%
train = pd.read_feather('../../data/feather/train_merged.feather')
test = pd.read_feather('../../data/feather/test_merged.feather')

# %%
# Site 0 remove missing meter readings from Site 0
train = train[~((train.site_id==0) & 
                (train.meter==0) & 
                (train.building_id <= 104) & 
                (train.timestamp < "2016-05-21"))]

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import notebook as tqdm
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# %%
X = train.drop(columns='meter_reading')
y = train['meter_reading'].map(np.log1p)

# %% [markdown]
# if the following error appears
#
# ```error: 'i' format requires -2147483648 <= number <= 2147483647```
#
# probably joblib multiprocessing fails to pickle data and send between processes, which means you have to stay with n_jobs=1 (till a solution is available)
#
# To change the n_jobs=1, do it the following way if you import the pipeline from another script
#
# `build_pipeline.full_pipeline.n_jobs = 1`

# %%
NFOLDS = 10
folds  = KFold(n_splits=NFOLDS, shuffle=False, random_state=123)

oof_preds = np.zeros((len(train), 1))
test_preds = np.zeros((len(test), 1))

for fold_, (train_index, val_index) in enumerate(folds.split(X)):
    train_X = X.iloc[train_index]
    val_X = X.iloc[val_index]
    train_y = y.iloc[train_index]
    val_y = y.iloc[val_index]
    
    lgbm = LGBMRegressor(metric='rmse')
    
    model_pipeline = Pipeline(steps=[('full_pipeline', build_pipeline.full_pipeline), 
                                      ('model', lgbm)])
    print(f'Model {fold_} fitting...')
    model_pipeline.fit(train_X, train_y)
    print(f'Model {fold_} fitted.')
    val_pred = np.expm1(model_pipeline.predict(val_X))
    
    rmse = np.sqrt(mean_squared_error(val_y, val_pred))
    
    print(f'Fold: {fold_} , RMSE: {rmse}')
    
    test_fold_pred = np.expm1(model_pipeline.predict(test))
    
    oof_preds[val_index,:] = val_pred.reshape((-1, 1))
    
    test_preds += test_fold_pred.reshape((-1, 1))
    
test_preds /= NFOLDS


# %%
def metric(y_pred, y_true):
    return np.sqrt(np.mean(np.power(np.log(y_pred + 1) - np.log(y_true + 1), 2)))


# %%
metric(np.clip(a = np.squeeze(oof_preds), a_min=0, a_max=None), np.expm1(y).values)

# %%
from datetime import datetime

# %%
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

p = Path('../../data/output/')

oof_prediction_file = p / f"oof_{timestamp}_{model_pipeline.steps[-1][1].__class__.__name__}.csv"
test_prediction_file = p / f"test_{timestamp}_{model_pipeline.steps[-1][1].__class__.__name__}.csv"

p.mkdir(parents=True, exist_ok=True)


# %%
oof_preds = np.clip(a = np.squeeze(oof_preds), a_min=0, a_max=None)
test_preds = np.clip(a = np.squeeze(test_preds), a_min=0, a_max=None)

submission = pd.DataFrame({'row_id':test.index, 'meter_reading':test_preds})
submission.to_csv(test_prediction_file, index=False)

oof_prediction = pd.DataFrame({'row_id':X.index, 'meter_reading':oof_preds})
oof_prediction.to_csv(oof_prediction_file, index=False)

# %%
X.site_id.unique()

# %%

# %%
from tqdm import tqdm_notebook as tqdm

# %%
oof_preds = np.zeros((len(train), 1))
test_preds = np.zeros((len(test), 1))

models = {}

for site_id in tqdm(range(16), desc='site_id'):
    
    X_site = X[X['site_id']==site_id]
    y_site = y.loc[X_site.index]
    test_site = test[test['site_id']==site_id]
    
    models[site_id] = []
    rmse_scores = []

    NFOLDS = 10
    folds  = KFold(n_splits=NFOLDS, shuffle=False, random_state=123)

    print(f"{NFOLDS} CV for site_id: {site_id}")
    
    for fold_, (train_index, val_index) in enumerate(folds.split(X_site)):
        train_X = X_site.iloc[train_index]
        val_X = X_site.iloc[val_index]
        train_y = y_site.iloc[train_index]
        val_y = y_site.iloc[val_index]

        lgbm = LGBMRegressor(metric='rmse')

        model_pipeline = Pipeline(steps=[('full_pipeline', build_pipeline.full_pipeline), 
                                          ('model', lgbm)])
  
        model_pipeline.fit(train_X, train_y)
        val_pred = np.expm1(model_pipeline.predict(val_X))

        # Scoring
        rmse = np.sqrt(mean_squared_error(val_y, val_pred))
        print(f'\tFold: {fold_} , RMSE: {rmse}')

        test_fold_pred = np.expm1(model_pipeline.predict(test.iloc[test_site.index]))
        oof_preds[val_index,:] = val_pred.reshape((-1, 1))
        test_preds[test_site.index] += test_fold_pred.reshape((-1, 1))
        
        models[site_id].append(model_pipeline)
        
        rmse_scores.append(rmse)
    
    print(f"{NFOLDS} CV for site_id: {site_id} fitted. RMSE: {np.array(rmse_scores).mean()}+/-{np.array(rmse_scores).std()} ")

test_preds /= NFOLDS


# %%
def metric(y_pred, y_true):
    return np.sqrt(np.mean(np.power(np.log(y_pred + 1) - np.log(y_true + 1), 2)))

metric(np.clip(a = np.squeeze(oof_preds), a_min=0, a_max=None), np.expm1(y).values)

# %%
