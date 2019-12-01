# %%
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.

import logging

#%%
def select_buildings_on_site(bldg_meter_recs, bldg_meta, site_number):
    """Given a site ID, select all building meter records matching this site.

    :param bldg_meter_recs:
    :param bldg_meta:
    :param site_number:
    :return:
    """
    select_bldg_ids = bldg_meta[bldg_meta['site_id'] == site_number]
    select_bldg_ids = select_bldg_ids.index
    sel = bldg_meter_recs[bldg_meter_recs['building_id'].isin(select_bldg_ids)]
    logging.info("Selected all {} buildings in site {}, {:0.1%} of data".format(
        len(select_bldg_ids), site_number, len(sel) / len(bldg_meter_recs) ))
    return sel

def get_building(df_meter_records, bldg_id):
    """
    Given the flat records for each meter-building-site

    Break out a summary energy profile DF for each building with the columns;
    Timestamp - Meter 1 - Meter 1 - Meter 1 - Meter 1
    Renames meter columns to proper energy sources
    """

    # Select the building records
    bldg_df = df_meter_records[df_meter_records.loc[:, 'building_id'] == bldg_id]
    bldg_df.drop('building_id', axis=1, inplace=True)
    logging.info("Building {} with {} records over {} meters".format(bldg_id, len(bldg_df), bldg_df['meter'].nunique()))

    # Pivot the records to meter-meter-meter-meter format
    bldg_df_pivot = bldg_df.pivot(index='timestamp', columns='meter', values='meter_reading')

    # For convenience, create empty columns for missing meters
    for meter_col in range(4):
        if meter_col not in bldg_df_pivot.columns:
            bldg_df_pivot[meter_col] = int(0)

    # bldg_df_pivot.memory_usage().sum() / 1024 ** 2 # Size in MB
    bldg_df_pivot.rename(columns={0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}, inplace=True)
    logging.info("{} to {}".format(bldg_df_pivot.index[0], bldg_df_pivot.index[-1]))
    assert bldg_df_pivot.index.is_monotonic_increasing

    return bldg_df_pivot

# for i in range(1000):
# bldg_id = i
# this_bldg = get_building(train_df, bldg_id)


#%%
def get_site_weather(weather_df, site_id):

    # Select the building records
    site_weather_df = weather_df[weather_df.loc[:, 'site_id'] == site_id]
    site_weather_df.drop('site_id', axis=1, inplace=True)
    logging.info("Site {} with {} records".format(site_id, len(site_weather_df)))
    site_weather_df.set_index('timestamp', drop=True, inplace=True)
    logging.info("{} to {}".format(site_weather_df.index[0], site_weather_df.index[-1]))
    assert site_weather_df.index.is_monotonic_increasing

    return site_weather_df

# site_id = 0
# for i in range(16):
#     r = get_site_weather(weather_train_df,i)
# r = weather_train_df.head()

#%%
