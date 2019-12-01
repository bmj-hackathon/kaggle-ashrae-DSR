# kaggle-ashrae-DSR

## DATA organization

Your local data folder should have
```
./raw
./feather
```

## Data leakage
All leaked data conforms to a common standard with reduced memory: 

```
Data columns (total 4 columns):
building_id              int16
meter                    int8
timestamp                datetime64[ns]
meter_reading_scraped    float32
```

- Site 0, UCF campus buildings
    - Complete data for all buildings, years
- Site 1, UCL campus buildings
    - Missing only one building
- Site 2, ASU campus
    - ?? 
- Site 4, Berkeley UCB campus
    - Missing some buildings
- Site 15, Cornell
    - Does not include 2016


## Psychrometric features
Both test and train weather have psychrometrics calculated. 




### Linking the DATA folder

`ln -s YOUR_DATA_PATH ./data`

i.e.;

`ln -s ~/DATA/ashrae-energy-prediction/data ./data`

## Environment management

### Jupyter Kernel
1. Activate the env
1. Ensure ipykernel is pip-installed
1. Install the kernel to jupyter

ipython kernel install --user --name=kaggle_ashrae_dsr
