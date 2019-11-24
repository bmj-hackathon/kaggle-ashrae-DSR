trfs.TemporalTransformer


sk.impute.SimpleImputer
sk.pipeline.FeatureUnion,

sk.preprocessing.QuantileTransformer
sk.preprocessing.KBinsDiscretizer
sk.preprocessing.OneHotEncoder
sk.preprocessing.StandardScaler


# categorical_features1 = ['site_id', 'meter', 'primary_use']
# categorical_features2 =
# numerical_features = ['square_feet', 'air_temperature', 'dew_temperature', 'wind_speed']
# temporal_features = ['timestamp']

categorical_pipeline1 = sk.pipeline.Pipeline(steps=[('cat_selector', trfs.FeatureSelector(['building_id'])),
                                       ('label_encoder', ce.one_hot.OneHotEncoder(handle_unknown='value'))])


this_df = train_df.head(10000)

categorical_pipeline1.steps

r = categorical_pipeline1.fit_transform(train_df)
