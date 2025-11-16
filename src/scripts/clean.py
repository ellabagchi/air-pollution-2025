# this script cleans the OMI NO2, OMI O3, or HCHO data(2005-2024) ---- ONLY USE THIS FOR NO2!!!!!!!!!!!!!!!!!!
# 1. extracts CloudScreenedTropNO2/ColumnAmountO3/ColumnAmountHCHO data and creates new lat/lon grid
# 2. filters lat and lon to my desired region
# 3. saves this cleaned data as a netCDF
import h5py
import numpy as np
import xarray as xr
import os

def extract_data(file_path):
    with h5py.File(file_path, 'r') as f:  # read the hdf5 file
        data_path = '/HDFEOS/GRIDS/OMI Total Column Amount HCHO/Data Fields/ColumnAmountHCHO'
        dset = f[data_path]

        # Just take the first step (assumes shape is (n_step, n_lat, n_lon))
        data = dset[0, :, :].astype("float32")  # make sure it's float so it can hold NaNs

        # ---- handle NaNs / fill values ----
        # Try the most common attribute names:
        fill_value = None
        for key in ["_FillValue", "MissingValue", "missing_value"]:
            if key in dset.attrs:
                fill_value = dset.attrs[key]
                break

        if fill_value is not None:
            # Replace fill values with NaN
            data = np.where(data == fill_value, np.nan, data)

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

    ds = xr.Dataset(
        {
            "HCHO": (["lat", "lon"], data)
        },
        coords={
            "lat": lat,
            "lon": lon
        }
    )

    return ds


def filter(ds):
    lat_max, lat_min = 43.125, 36.375 
    lon_min, lon_max = -83.125, -70.000 
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

    for year in range(2005,2025):  # inclusive of 2024 for (2005, 2025)
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
