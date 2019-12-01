logging.info(" *** Step 3: Feature engineering *** ".format())

#%%
feature_union = list()

#%% Time features
feature_union.append(sk.pipeline.Pipeline(steps=[('time', trfs.TimeFeatures('timestamp')), ]))


#%% Categorical features to OneHot
categorical_features1 = ['meter', 'primary_use']

feature_union.append(sk.pipeline.Pipeline(steps=[
    ('onehot_encoder', trfs.MyOneHotDF(categorical_features1))
]))

#%%
for i, pipe in enumerate(feature_union):
    print(i)
    print("\t",pipe.named_steps)


#%%


#%%

this_pipe = sk.pipeline.Pipeline(steps=[
    ('cat_selector1', trfs.MyColumnSelector(categorical_features1)),
    ('onehot_encoder', sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore'))
])
res = this_pipe.fit_transform(df_train.copy())

# res2 = pd.get_dummies(df_train[categorical_features1].copy(),prefix=['country'])
res2 = pd.get_dummies(df_train[categorical_features1].copy(),dummy_na=True)


#%%

res = pipe_time_features.fit_transform(df_train.copy())

preprocessing_pipeline = sk.pipeline.FeatureUnion(transformer_list=[
    ('Get time features', pipe_time_features),
    # ('', numerical_pipeline),
    # ('', temporal_pipeline),
], verbose=True)
# ], n_jobs=-2, verbose=True)

res = preprocessing_pipeline.fit_transform(df_train.copy())


full_pipeline = sk.pipeline.Pipeline(steps=[('features', preprocessing_pipeline)])

res = full_pipeline.fit_transform(df_train.copy())

