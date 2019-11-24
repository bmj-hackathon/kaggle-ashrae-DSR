import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn as sk

def timeit(method):
    """ Decorator to time execution of transformers
    :param method:
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print("\t {} {:2.1f}s".format(method.__name__, (te - ts)))
        return result

    return timed

#%%
class TransformerLog():
    """Add a .log attribute for logging
    """
    @property
    def log(self):
        return "Transformer: {}".format(type(self).__name__)

#%%
class MultipleToNewFeature(sk.base.BaseEstimator, sk.base.TransformerMixin, TransformerLog):
    """Given a list of column names, create a new column in the df. New column defined by func.
    """

    def __init__(self, selected_cols, new_col_name, func):
        self.selected_cols = selected_cols
        self.new_col_name = new_col_name
        self.func = func

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, df, y=None):
        df[self.new_col_name] = df.apply(self.func, axis=1)
        print(self.log, "{}({}) -> ['{}']".format(self.func.__name__, self.selected_cols, self.new_col_name))
        return df

#%%
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self._feature_names]
        except KeyError:
            cols_error = list(set(self._feature_names) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)


class TemporalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self._column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert pd.api.types.is_datetime64_any_dtype(X[self._column])

        out = pd.DataFrame()

        out['hour'] = X[self._column].dt.hour
        out['month'] = X[self._column].dt.month
        # X['week'] = X[self._column].dt.week
        out['weekday'] = X[self._column].dt.weekday
        out['quarter'] = X[self._column].dt.quarter
        out['weekend'] = np.where(X[self._column].dt.weekday > 4, 1, 0)
        return out
