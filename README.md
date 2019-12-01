# kaggle-ashrae-DSR

## DATA organization

Your local data folder should have
```
./raw
./feather
```

## Data leakage
- Site 0
    - Complete data for all years
    

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
