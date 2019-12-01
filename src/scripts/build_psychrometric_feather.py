# Given the full original site dataframe, augment with psychrometric air properties
# sea_level_pressure_pa Interpolate pressure, convert to Pascals
# Add:
# wet-bulb [C], relative humidity [0-1], humidity ratio [kg/kg], sensbile/latent heats [kJ/kg]

import psychrolib as psychrometric
psychrometric.SetUnitSystem(psychrometric.SI)

#%% Weather

def get_wet_bulb(air_temp, dew_temp, pressure):
    """Utility function, passes through to psychrolib

    Not sure why this is needed, but sometimes the pure function breaks during convergence??
    :param air_temp:
    :param dew_temp:
    :param pressure:
    :return:
    """
    try:
        return psychrometric.GetTWetBulbFromTDewPoint(air_temp, dew_temp, pressure)
    except:
        return np.NaN

def get_rel_hum(air_temp, dew_temp):
    """Utility function, passes through to psychrolib

    Not sure why this is needed, but sometimes the pure function breaks during convergence??
    """
    try:
        return psychrometric.GetRelHumFromTDewPoint(air_temp, dew_temp)
    except:
        return np.NaN

def add_psychrometric_weather(original_weather_df):
    """Given a properly organized weather dataframe, add moist air (psychrometric) properties

    :param this_weather_df:
    :return: this_weather_df
    """
    this_weather_df = original_weather_df.copy()
    # Interpolate the pressure
    count_nan = this_weather_df['sea_level_pressure'].isna().sum()
    if this_weather_df['sea_level_pressure'].isna().sum():
        this_weather_df['sea_level_pressure'].interpolate(method='linear', inplace=True)
        logging.info("Psychrometry: Interpolated sea level pressure, {} missing values".format(count_nan))
    if this_weather_df['sea_level_pressure'].isna().sum():
        this_weather_df['sea_level_pressure'] = 101300
        logging.info("Psychrometry: WARNING no sea level pressure data, setting to 101.3 kPa! {} missing values".format(count_nan))
    this_weather_df['sea_level_pressure_pa'] = this_weather_df['sea_level_pressure'] / 10 * 1000

    # Get the other psychrometric properties
    this_weather_df['wet_bulb_temp'] = this_weather_df.apply(lambda x: get_wet_bulb(x['air_temperature'], x['dew_temperature'], x['sea_level_pressure_pa']), axis=1)
    logging.info("Psychrometry: Added wet-bulb [C]".format())
    this_weather_df['rel_hum'] = this_weather_df.apply(lambda x: get_rel_hum(x['air_temperature'], x['dew_temperature']), axis = 1)
    logging.info("Psychrometry: Added relative humidity [0-1]".format())
    this_weather_df['hum_ratio'] = this_weather_df.apply(lambda x: psychrometric.GetHumRatioFromRelHum(x['air_temperature'], x['rel_hum'], x['sea_level_pressure_pa']), axis=1)
    logging.info("Psychrometry: Added Humidity Ratio [kg/kg]".format())
    this_weather_df['sensible_heat'] = this_weather_df.apply(lambda x: psychrometric.GetDryAirEnthalpy(x['air_temperature']), axis=1)
    this_weather_df['latent_heat'] = this_weather_df.apply(lambda x: psychrometric.GetSatAirEnthalpy(x['air_temperature'], x['sea_level_pressure_pa']), axis=1)
    logging.info("Psychrometry: Added sensible/latent heats [kJ/kg]".format())
    return this_weather_df

# site_id = 5
# weather_df = get_site_weather(weather_train_df, site_id)
# weather_df = add_psychrometric_weather(weather_df)

#%%
# Write each site to feather
if 0:
    for site_id in range(16):
        # site_id = 1
        weather_df = get_site_weather(weather_train_df, site_id)
        weather_df = add_psychrometric_weather(weather_df)
        fname = 'weather_psychrometric_site_{}.feather'.format(site_id)
        weather_df.reset_index().to_feather(fname)
        logging.info("Wrote {}".format(fname))
        r = pd.read_feather(fname)

#%% TRAIN Write entire extra weather features to new DF
if 0:
    path_psychro = SETTINGS.data.path_data_root / 'feature_psychrometric'
    assert path_psychro.exists()
    original_weather_cols = weather_train_df.columns.to_list()
    weather_psychro_df = add_psychrometric_weather(weather_train_df)
    assert not weather_psychro_df is weather_train_df
    weather_psychro_df.drop(original_weather_cols, axis=1, inplace=True)
    weather_psychro_df.to_feather(path_psychro / 'weather_train_psychrometric.feather')

#%% TEST Write entire extra weather features to new DF
if 0:
    path_psychro = SETTINGS.data.path_data_root / 'feature_psychrometric'
    assert path_psychro.exists()
    original_weather_cols = weather_test_df.columns.to_list()
    weather_psychro_df = add_psychrometric_weather(weather_test_df)
    assert not weather_psychro_df is weather_test_df
    weather_psychro_df.drop(original_weather_cols, axis=1, inplace=True)

    weather_psychro_df.to_feather(path_psychro / 'weather_test_psychrometric.feather')
