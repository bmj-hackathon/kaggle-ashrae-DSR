from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import QuantileTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler, RobustScaler
import category_encoders as ce
from src.utils.ashrae_transformers import *

categorical_features1 = ['meter', 'primary_use']
categorical_features2 = ['building_id']
numerical_features = ['square_feet', 'air_temperature', 'dew_temperature', 'wind_speed']
temporal_features = ['timestamp']


# Build pipelines
categorical_pipeline1 = Pipeline(steps=[('cat_selector1', FeatureSelector(categorical_features1)),
                                        ('onehot_encoder', OneHotEncoder(handle_unknown='ignore'))])

categorical_pipeline2 = Pipeline(steps=[('cat_selector2', FeatureSelector(categorical_features2)),
                                        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))])

numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(numerical_features)),
                                     ('imputer', SimpleImputer(strategy='median')),
                                     ('std_scaler', StandardScaler())])

temporal_pipeline = Pipeline(steps=[('time_selector', FeatureSelector(temporal_features)),
                                    ('temporal_features', TemporalTransformer('timestamp')),
                                    ('std_scaler', StandardScaler())])

# Combine pipelines
preprocessing_pipeline = FeatureUnion(transformer_list=[('categorical_pipeline1', categorical_pipeline1),
                                                        ('categorical_pipeline2', categorical_pipeline2),
                                                        ('numerical_pipeline', numerical_pipeline),
                                                        ('temporal_pipeline', temporal_pipeline)],
                                      n_jobs=-1,
                                      verbose=False)

# Combine pipeline as named pipeline
full_pipeline = Pipeline(steps=[('features', preprocessing_pipeline)])



