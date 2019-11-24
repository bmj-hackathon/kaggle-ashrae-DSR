
# %%
print ('oof score meter0 =', np.sqrt(oof0))
print ('oof score meter1 =', np.sqrt(oof1))
print ('oof score meter2 =', np.sqrt(oof2))
print ('oof score meter3 =', np.sqrt(oof3))
print ('oof score total  =', np.sqrt(oof_total / len(train_df)))

# %% [markdown]
# # Prediction on test data

# %%
del train_df, weather_train_df, building_meta_df
gc.collect()

# %% {"_kg_hide-input": true}
def create_X(test_df, target_meter):
    target_test_df = test_df[test_df['meter'] == target_meter]
    target_test_df = target_test_df.merge(building_meta_df, on='building_id', how='left')
    target_test_df = target_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
    X_test = target_test_df[feature_cols + category_cols]
    return X_test


# %% {"_kg_hide-input": true}
def pred(X_test, models, batch_size=1000000):
    iterations = (X_test.shape[0] + batch_size -1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, model in enumerate(models):
        print(f'predicting {i}-th model')
        for k in tqdm.tqdm(range(iterations)):
            y_pred_test = model.predict(X_test[k * batch_size:(k + 1) * batch_size], num_iteration=model.best_iteration)
            y_test_pred_total[k * batch_size:(k + 1) * batch_size] += y_pred_test

    y_test_pred_total /= len(models)
    return y_test_pred_total


# %%
# %%time
X_test = create_X(test_df, target_meter=0)
gc.collect()

y_test0 = pred(X_test, models0)

sns.distplot(y_test0)

del X_test
gc.collect()

# %%
# %%time
X_test = create_X(test_df, target_meter=1)
gc.collect()

y_test1 = pred(X_test, models1)
sns.distplot(y_test1)

del X_test
gc.collect()

# %%
# %%time
X_test = create_X(test_df, target_meter=2)
gc.collect()

y_test2 = pred(X_test, models2)
sns.distplot(y_test2)

del X_test
gc.collect()

# %%
X_test = create_X(test_df, target_meter=3)
gc.collect()

y_test3 = pred(X_test, models3)
sns.distplot(y_test3)

del X_test
gc.collect()

# %%
sample_submission.loc[test_df['meter'] == 0, 'meter_reading'] = np.expm1(y_test0)
sample_submission.loc[test_df['meter'] == 1, 'meter_reading'] = np.expm1(y_test1)
sample_submission.loc[test_df['meter'] == 2, 'meter_reading'] = np.expm1(y_test2)
sample_submission.loc[test_df['meter'] == 3, 'meter_reading'] = np.expm1(y_test3)

# %%
sample_submission.to_csv(SETTINGS.data.path_output / 'submission.csv', index=False, float_format='%.4f')

# %% [markdown]
# # Replace to UCF data
if 0:
    # %%
    leak_score = 0

    leak_df = pd.read_pickle(ucf_root / 'site0.pkl')
    leak_df['meter_reading'] = leak_df.meter_reading_scraped
    leak_df.drop(['meter_reading_original', 'meter_reading_scraped'], axis=1, inplace=True)
    leak_df.fillna(0, inplace=True)
    leak_df = leak_df[leak_df.timestamp.dt.year > 2016]
    leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0  # remove large negative values

    sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0

    for bid in leak_df.building_id.unique():
        temp_df = leak_df[(leak_df.building_id == bid)]
        for m in temp_df.meter.unique():
            v0 = sample_submission.loc[(test_df.building_id == bid) & (test_df.meter == m), 'meter_reading'].values
            v1 = temp_df[temp_df.meter == m].meter_reading.values

            leak_score += sk.metrics.mean_squared_error(np.log1p(v0), np.log1p(v1)) * len(v0)

            sample_submission.loc[(test_df.building_id == bid) & (test_df.meter == m), 'meter_reading'] = temp_df[
                temp_df.meter == m].meter_reading.values

    # %%
    if not SETTINGS.control.debug:
        sample_submission.to_csv('submission_ucf_replaced.csv', index=False, float_format='%.4f')

    # %%
    sample_submission.head()

    # %%
    np.log1p(sample_submission['meter_reading']).hist()

    # %% [markdown]
    # # UCF score

    # %%
    print('UCF score = ', np.sqrt(leak_score / len(leak_df)))

    # %%
    plot_feature_importance(models0[1])

    # %%
    plot_feature_importance(models1[1])

    # %%
    plot_feature_importance(models2[1])

    # %%
    plot_feature_importance(models3[1])

    # %% [markdown]
    # # References
    #
    # These kernels inspired me to write this kernel, thank you for sharing!
    #
    #  - https://www.kaggle.com/rishabhiitbhu/ashrae-simple-eda
    #  - https://www.kaggle.com/isaienkov/simple-lightgbm
    #  - https://www.kaggle.com/ryches/simple-lgbm-solution
