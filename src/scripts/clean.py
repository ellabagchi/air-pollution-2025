# this script cleans the OMI NO2 data or OMI O3 data(2005-2024)
# 1. extracts CloudScreenedTropNO2/ColumnAmountO3 data and creates new lat/lon grid
# 2. filters lat and lon to my desired region
# 3. saves this cleaned data as a netCDF
import h5py
import numpy as np
import xarray as xr
import os

def extract_data(file_path):
    with h5py.File(file_path, 'r') as f: # read the hdf5 file
        # ***change variable below
        #data_path = '/HDFEOS/GRIDS/ColumnAmountNO2/Data Fields/ColumnAmountNO2TropCloudScreened' # uncomment for NO2
        #data_path = '/HDFEOS/GRIDS/ColumnAmountO3/Data Fields/ColumnAmountO3' # for O3
        #data = f[data_path][()].copy() # for no2 and o3

        data_path = '/HDFEOS/GRIDS/OMI Total Column Amount HCHO/Data Fields/ColumnAmountHCHO' # for HCHO
        data = f[data_path][0].copy() # for hcho -- hopefully takes the first step/array in col. amt. HCHO

        # below-- reconstruct lat/lon grid because I couldn't access the lat/lon in original file
        n_lat, n_lon = data.shape 
        lat_start = -90 + 0.25 / 2
        lat_end = 90 - 0.25 / 2
        lon_start = -180 + 0.25 / 2
        lon_end = 180 - 0.25 / 2

        latitudes = np.linspace(lat_start, lat_end, n_lat)
        longitudes = np.linspace(lon_start, lon_end, n_lon)

        lat = latitudes
        lon = longitudes

    ds = xr.Dataset( # create a new ds for the data
        {
            'HCHO': (['lat', 'lon'], data) # ***change variable here 
        },
        coords={
            'lat': lat,
            'lon': lon
        }
    )

    return ds

def filter(ds):
    lat_max, lat_min = 43.125, 36.375 
    lon_min, lon_max = -81.125, -72.875 
    filtered_data = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    return filtered_data

def main():
    # ***change variables below
    #input_base_dir = "/home/ellab/air_pollution/src/data/new_no2" # input directory for NO2
    #input_base_dir = "/home/ellab/air_pollution/src/data/new_o3" # O3
    input_base_dir = "/home/ellab/air_pollution/src/data/new_hcho" # HCHO

    #output_base_dir = "/home/ellab/air_pollution/src/data/clean_no2" # output directory for NO2
    #output_base_dir = "/home/ellab/air_pollution/src/data/clean_o3" # O3
    output_base_dir = "/home/ellab/air_pollution/src/data/clean_hcho" # HCHO

    for year in range(2005,2006):  # inclusive of 2024 for (2005, 2025)
        year_dir = os.path.join(input_base_dir, str(year)) 
        if not os.path.isdir(year_dir):
            continue  # skip if year folder doesn't exist

        # create matching year folder in output
        output_year_dir = os.path.join(output_base_dir, str(year))
        os.makedirs(output_year_dir, exist_ok=True)

        for filename in os.listdir(year_dir):
            if filename.endswith(".he5"):
                file_path = os.path.join(year_dir, filename)
                try:
                    ds = extract_data(file_path) # extract
                    filtered_no2 = filter(ds) # filter

                    # Output filename
                    output_filename = filename.replace(".he5", "_clean.nc")
                    output_path = os.path.join(output_year_dir, output_filename)
                    # Save filtered dataset
                    filtered_no2.to_netcdf(output_path)
                    print(f"Saved: {output_path}")

                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    main()
