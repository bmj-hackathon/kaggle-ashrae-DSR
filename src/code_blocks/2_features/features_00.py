logging.info(" *** Step 3: Feature engineering *** ".format())

df_train.info()
df_train

# mapper = DataFrameMapper([
#      # ('pet', sk.preprocessing.LabelBinarizer()),
#      # (['children'], sk.preprocessing.StandardScaler())
#  ])
pipeline = sk.pipeline.Pipeline(steps=[('1', trfs.FeatureSelector([])), ])

pipeline.fit_transform(df_train.copy())