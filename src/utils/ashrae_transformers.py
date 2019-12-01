import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import time
import logging
from deprecated import deprecated
import pandas.api.types as ptypes

__all__ = ['TypeSelector', 'FeatureSelector', 'TemporalTransformer']

def dump_func_name(func):
    def echo_func(*func_args, **func_kwargs):
        print('')
        print('Start func: {}'.format(func.__name__))
        return func(*func_args, **func_kwargs)
    return echo_func

def timeit(method):
    """Decorator to time execution of transformers
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
            logging.info("\t {} method took {:2.1f}s".format(method.__name__, (te - ts)))
        return result

    return timed


class UtilityMixin:
    @property
    def log(self):
        return "Transformer: {}".format(type(self).__name__)

    def _check_columns_exist(self, df):
        self._assert_df(df)

        # Must have columns specified
        assert self.columns, "No columns specified!"

        # Ensure columns in a list()
        if isinstance(self.columns, str):
            raise ValueError("Columns must be specified as a list, you specified a string: {}".format(self.columns))
            # self.columns = list(self.columns)

        self._assert_df(df)

        # Now check each column against the dataframe
        missing_columns = set(self.columns) - set(df.columns)
        if len(missing_columns) > 0:
            missing_columns_ = ','.join(col for col in missing_columns)
            raise KeyError(
                "Keys are missing in the record: {}, columns required:{}".format(missing_columns_, self.columns))

    def _assert_df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input is not a pandas DataFrame it's a {}".format(type(df)))


class TransformerLog():
    """Add a .log attribute for logging
    """
    @property
    def log(self):
        return "{} transform".format(type(self).__name__)


class MultipleToNewFeature(BaseEstimator, TransformerMixin, UtilityMixin):

    def __init__(self, selected_cols, new_col_name, func):
        """Given a list of column names, create one new column in the df. New column defined by func.

        :param selected_cols: The list of columns to pass into func
        :param new_col_name: The newly created column name
        :param func: The function to be applied to each row
        """
        self.columns = selected_cols
        self.new_col_name = new_col_name
        self.func = func


    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, df, y=None):
        self._assert_df(df)
        self._check_columns_exist(df)
        df[self.new_col_name] = df.apply(self.func, axis=1)
        logging.info(self.log, "{}({}) -> ['{}']".format(self.func.__name__, self.columns, self.new_col_name))
        return df


class TypeSelector(BaseEstimator, TransformerMixin, TransformerLog):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])

@deprecated(reason="Deprecated to MyColumnSelector")
class FeatureSelector(BaseEstimator, TransformerMixin, TransformerLog):
    def __init__(self, feature_names):
        self._feature_names = feature_names


    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, X, y=None):
        logging.info("{} - {}".format(self.log, self._feature_names))
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self._feature_names]
        except KeyError:
            cols_error = list(set(self._feature_names) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)



class MyColumnSelector(BaseEstimator, TransformerMixin, TransformerLog, UtilityMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, df, y=None):
        logging.info("{} - {}".format(self.log, self.columns))

        self._assert_df(df)
        self._check_columns_exist(df)

        return df[self.columns]

@deprecated(reason="Deprecated to TimeFeatures")
class TemporalTransformer(BaseEstimator, TransformerMixin, TransformerLog):
    def __init__(self, column):
        self._column = column

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, X, y=None):
        assert pd.api.types.is_datetime64_any_dtype(X[self._column])

        out = pd.DataFrame()

        out['hour'] = X[self._column].dt.hour
        out['month'] = X[self._column].dt.month
        # X['week'] = X[self._column].dt.week
        out['weekday'] = X[self._column].dt.weekday
        out['quarter'] = X[self._column].dt.quarter
        out['weekend'] = np.where(X[self._column].dt.weekday > 4, 1, 0)
        out['early_morning'] = np.where(np.logical_and(out['hour'] > 6,
                                                       out['hour'] < 8),
                                        1, 0)
        out['morning'] = np.where(np.logical_and(out['hour'] > 8,
                                                 out['hour'] < 12),
                                  1, 0)
        out['afternoon'] = np.where(np.logical_and(out['hour'] > 12,
                                                   out['hour'] < 16),
                                    1, 0)
        out['evening'] = np.where(np.logical_and(out['hour'] > 16,
                                                 out['hour'] < 20),
                                  1, 0)
        out['night'] = np.where(np.logical_and(out['hour'] > 20,
                                               out['hour'] < 6),
                                1, 0)
        out['monday_morning'] = np.where(np.logical_and(out['hour'] < 7,
                                                        out['weekday'] == 0),
                                         1, 0)
        out['friday_evening'] = np.where(np.logical_and(out['hour'] > 16,
                                                        out['weekday'] == 4),
                                         1, 0)
        return out

class TimeFeatures(BaseEstimator, TransformerMixin, TransformerLog, UtilityMixin):
    def __init__(self, column_name):
        assert type(column_name) == str
        # Only one column
        self.columns = [column_name]

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, df, y=None):
        self._assert_df(df)
        self._check_columns_exist(df)

        # Only one column for this transformer
        this_col = self.columns[0]
        assert pd.api.types.is_datetime64_any_dtype(df[this_col])

        df_return = pd.DataFrame()

        df_return['hour'] = df[this_col].dt.hour
        df_return['month'] = df[this_col].dt.month
        df_return['week'] = df[this_col].dt.week
        df_return['weekday'] = df[this_col].dt.weekday
        df_return['quarter'] = df[this_col].dt.quarter
        df_return['weekend'] = np.where(df[this_col].dt.weekday > 4, 1, 0)
        # df_return['early_morning'] = np.where(np.logical_and(df_return['hour'] > 6, df_return['hour'] < 8), 1, 0)
        # df_return['morning'] = np.where(np.logical_and(df_return['hour'] > 8, df_return['hour'] < 12), 1, 0)
        # df_return['afternoon'] = np.where(np.logical_and(df_return['hour'] > 12, df_return['hour'] < 16), 1, 0)
        # df_return['evening'] = np.where(np.logical_and(df_return['hour'] > 16, df_return['hour'] < 20), 1, 0)
        # df_return['night'] = np.where(np.logical_and(df_return['hour'] > 20, df_return['hour'] < 6), 1, 0)
        # df_return['monday_morning'] = np.where(np.logical_and(df_return['hour'] < 7, df_return['weekday'] == 0), 1, 0)
        # df_return['friday_evening'] = np.where(np.logical_and(df_return['hour'] > 16, df_return['weekday'] == 4), 1, 0)
        return df_return

@deprecated(reason="Don't use DataFrame Mapper class...")
class DFMapperTimeFeatures(BaseEstimator, TransformerMixin, TransformerLog, UtilityMixin):
    def __init__(self):
        pass
        # assert type(column_name) == str
        # Only one column
        # self.columns = [column_name]

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, X, y=None):
        # self._assert_df(df)
        # self._check_columns_exist(df)

        # Only one column for this transformer
        # this_col = self.columns[0]
        # assert pd.api.types.is_datetime64_any_dtype(df[this_col])
        print(X.shape)
        df_return = pd.DataFrame()

        df_return['hour'] = X.dt.hour
        df_return['month'] = X.dt.month
        df_return['week'] = X.dt.week
        df_return['weekday'] = X.dt.weekday
        df_return['quarter'] = X.dt.quarter
        df_return['weekend'] = np.where(X.dt.weekday > 4, 1, 0)
        return df_return

#%%
class MyOneHotDF(BaseEstimator, TransformerMixin, TransformerLog, UtilityMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    @timeit
    def transform(self, df, y=None):
        self._assert_df(df)
        self._check_columns_exist(df)

        df_subset = df[self.columns].copy()

        # Need to iterate over columns to process non-object/categorical dtypes
        df_return = pd.DataFrame()
        for col in df_subset.columns:
            # Force encode numeric and other types
            if not ptypes.is_string_dtype(df_subset[col]):
                this_df = df_subset[[col]].astype(str)
            else:
                this_df = df_subset[[col]]

            this_df = pd.get_dummies(this_df)
            # this_df.drop(col, axis=1,inplace=True)
            df_return = pd.concat([df_return, this_df], axis=1)

        return df_return