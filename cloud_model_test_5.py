from functions import get_patmos_s3_files 
import xarray as xr
from skimage import filters
import pandas as pd
import s3fs
import h5netcdf 
import netCDF4
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import fsspec
import zarr
import boto3
#from memory_profiler import profile

#@profile
def run_cloud_model(year, month, day, satellite, ascdes):
    """
    Process data and return a list.

    Args:
        year (int):  Year
        month (int): Month of year: range = [1,12]
        day (int):   Day of month range: [1-31]
        satellite (str): NOAA satellite name. E.g., 'NOAA-14'
        ascdec (str):    Should either be 'asc' or 'des'

    Returns:
        URLs: Return the s3 list of URLs 
    """
    
    model_path='model/xgboost_model_2.pkl'

    name=f"{year}{month:02d}{day:02d}"

    print (f"Running :: {name}")
    
    URLs = get_patmos_s3_files(year, month, day, satellite, ascdes)

    if len(URLs[0]) == 0:
        print(f"On day={name} no retrieval since yesterday is missing")
        return
    if len(URLs[1]) == 0:
        print(f"On day={name} no retrieval since today is missing")
        return
    if len(URLs[2]) == 0:
        print(f"On day={name} no retrieval since tomorrow is missing")
        return

    # Open and read in the ascending and descending file 
    fs = s3fs.S3FileSystem(anon=True) # or anon=False to use default credentials

    ds1= xr.open_dataset(fs.open(URLs[0], 'rb'))
    ds2= xr.open_dataset(fs.open(URLs[1], 'rb'))
    ds3= xr.open_dataset(fs.open(URLs[2], 'rb'))

    x1=1
    x2=3600
    y1=100
    y2=1700
    
    orig_shape=(y2-y1, x2-x1)

    print(f"On day={name} Repackaging input data")

    t2       = ds2['temp_11_0um_nom'].isel(           longitude=slice(x1, x2), latitude=slice(y1, y2), time=0).data
    # Assuming your 2D array is stored in a variable called 'brightness_temperatures'
####ir_edges_sobel = 

    #this is the ONLY variable at the original resolution of 1800x3600
    cf2      = ds2['cloud_fraction']
    #ir2_nans          = cf2.isnull()
    nan_mask = np.isnan(ds2['temp_11_0um_nom'].values)
    
    print(f"On day={name} Adding a few more features")
    # Convert the combined array into a DataFrame
    df_final = pd.DataFrame(dict(zip(['cf', 't1', 't2', 't3', 'tclr', 'sobel', 'snoice', 'sfc','t21','t23','dt'], 
                                      [ds2['cloud_fraction'].isel(              longitude=slice(x1, x2), latitude=slice(y1, y2), time=0).data.flatten(), 
                                       ds1['temp_11_0um_nom'].isel(             longitude=slice(x1, x2), latitude=slice(y1, y2), time=0).data.flatten(),
                                       t2.flatten(), 
                                       ds3['temp_11_0um_nom'].isel(             longitude=slice(x1, x2), latitude=slice(y1, y2), time=0).data.flatten(), 
                                       ds2['temp_11_0um_nom_clear_sky'].isel(   longitude=slice(x1, x2), latitude=slice(y1, y2), time=0).data.flatten(), 
                                       filters.sobel(t2).flatten(),
                                       ds2['snow_class'].isel(                  longitude=slice(x1, x2), latitude=slice(y1, y2), time=0).data.flatten(),
                                       ds2['land_class'].isel(                  longitude=slice(x1, x2), latitude=slice(y1, y2), time=0).data.flatten(),
                                       t2.flatten()-ds1['temp_11_0um_nom'].isel(longitude=slice(x1, x2), latitude=slice(y1, y2), time=0).data.flatten(),
                                       t2.flatten()-ds3['temp_11_0um_nom'].isel(longitude=slice(x1, x2), latitude=slice(y1, y2), time=0).data.flatten(),
                                       ds2['temp_11_0um_nom_clear_sky'].isel(   longitude=slice(x1, x2), latitude=slice(y1, y2), time=0).data.flatten()-t2.flatten()
                     ])))
    
    #binarize snowice: missing data(0) becomes -1, no snoice bcomes 0 and presence of snow or ice becomes 1
    df_final['snoice'] = df_final['snoice'].replace({0:-1, 1:0, 2:1, 3:1})
    
    #Make fewer surface categories:
    ##shallow ocean=0,land=1,coastline=2,shallow inland water=3,ephemeral water=4,deep inland water=5,moderate ocean=6,deep ocean=7
    ##shallow ocean=0,land=1,coastline=2,shallow inland water=1,ephemeral water=1,deep inland water=0,moderate ocean=0,deep ocean=0
    df_final['sfc'] = df_final['sfc'].replace({3:1, 4:1, 5:0, 6:0, 7:0})
    # new meaning: 0 = permamnent water, 1 = land (land or sometimes water), 2 = coastline   
    
    df_final['cfbin'] = (df_final['cf'] > 0.5).astype(int)
    df_final = df_final.drop(columns=['cf'])

    print(df_final.head())
    
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    print(f"On day={name} Run the model")
    # Make predictions on the data
#    X_all = df_final.drop(columns=['cf', 'cfbin'])
#    y_pred = loaded_model.predict(X_all)
    y_pred = loaded_model.predict(df_final.drop(columns=['cfbin']))

    print(f"On day={name} Reshape result")
    #reshape predicted data
    y_pred_2d = y_pred.reshape(orig_shape)
    y_pred_2d.shape
#    y_pred[t2_nans] = -1
    
    #save resulting Cloud Fraction Prediction to the file

    print(f"On day={name} Convert output to Xarray DataSet")
    #will write out 3 values:
    # 1. Original cloud flag - cf2 is already a zarray DataArray
    # 2. Model Cloud Flag --- needs to be converted to DataArray
    cf2_model = np.full_like(cf2[:,:,:],-1) #set missing values to -1
    cf2_model[0,y1:y2,x1:x2] = y_pred_2d
    
    #assign -1 to cf2_model where ir2 (t112) is nan
    cf2_model[nan_mask] = -1
    
    # model_out = xr.DataArray(cf2_model, 
    #                          dims=cf2.dims, 
    #                          coords=cf2.coords, 
    #                          name='cld_fraction_ml')

    # Combine DataArrays into a DataSet
    dataset = xr.Dataset({'cld_frac': cf2, 'cld_frac_model': xr.DataArray(cf2_model, 
                             dims=cf2.dims, 
                             coords=cf2.coords, 
                             name='cld_fraction_ml')})
    
    # Create the filename string
    filename = f"{year}/ML_Cld_Frac_{year}{month:02d}{day:02d}_{satellite}_{ascdes}.zarr"

    flag_s3_write=True

    if flag_s3_write:
        s3_bucket = 'your-output-bucket'
        s3_key = filename
        # Create a file-like object representing the S3 location
        s3_url = f's3://{s3_bucket}/{s3_key}'
        print(f"On day={name} Writing to {s3_url}")
        s3_data = fsspec.get_mapper(s3_url)

        # Write the DataArray to the Zarr file in the S3 bucket
        dataset.to_zarr(s3_data, mode='w')
    else:
        # Write DataSet to a Zarr file
        print(f"On day={name} Write result to local file: "+filename)
        dataset.to_zarr(filename)


if __name__ == "__main__":
    year=1999
    month=10
    day=10
    satellite='NOAA-14'
    ascdes='asc'
    run_cloud_model(year, month, day, satellite, ascdes)