#%%
def test_add_rows(row):
    return row['building_id'] + row['meter']

this_pipeline = sk.pipeline.Pipeline([
        ('feature: Adding rows', MultipleToNewFeature(['building_id','meter'], 'Pure Breed', test_add_rows)),
        ])

logging.info("Created pipeline:")
for i, step in enumerate(this_pipeline.steps):
    print(i, step[0], step[1].__str__())

df_trf = this_pipeline.fit_transform(train_df)
logging.info("Pipeline complete. {} new columns.".format(len(df_all.columns) - len(original_cols)))

#%%



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
