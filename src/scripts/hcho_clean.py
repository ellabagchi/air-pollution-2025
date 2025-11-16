# this script cleans the OMI HCHO data (2005-2024)
# 1. extracts ColumnAmountHCHO data + true lat/lon grid from the HDF5
# 2. converts fill/missing values to NaNs
# 3. filters to my desired region
# 4. saves cleaned data as a netCDF

import h5py
import numpy as np
import xarray as xr
import os

def extract_data(file_path):
    base = "/HDFEOS/GRIDS/OMI Total Column Amount HCHO"

    with h5py.File(file_path, "r") as f:
        # ---- HCHO data ----
        hcho_dset = f[f"{base}/Data Fields/ColumnAmountHCHO"]

        # Take first step: assumes shape (n_step, n_y, n_x)
        data = hcho_dset[0, :, :].astype("float32")

        # ---- Convert fill / missing values to NaN ----
        fill_value = None
        for key in ["_FillValue", "MissingValue", "missing_value"]:
            if key in hcho_dset.attrs:
                fill_value = hcho_dset.attrs[key]
                # sometimes stored as 0-d array
                if np.ndim(fill_value) > 0:
                    fill_value = fill_value[0]
                break

        if fill_value is not None:
            data = np.where(data == fill_value, np.nan, data)

        # ---- True latitude / longitude ----
        # Try common paths: some files put Latitude/Longitude under Data Fields,
        # some directly under the grid group.
        try:
            lat_dset = f[f"{base}/Data Fields/Latitude"]
            lon_dset = f[f"{base}/Data Fields/Longitude"]
        except KeyError:
            lat_dset = f[f"{base}/Latitude"]
            lon_dset = f[f"{base}/Longitude"]

        lat = lat_dset[0, :, :].astype("float32")
        lon = lon_dset[0, :, :].astype("float32")

    # Use "y", "x" as index dims; lat/lon as 2-D coordinate variables
    ds = xr.Dataset(
        data_vars={
            "HCHO": (["y", "x"], data)
        },
        coords={
            "lat": (["y", "x"], lat),
            "lon": (["y", "x"], lon),
        }
    )

    return ds


def filter(ds):
    # region bounds
    lat_max, lat_min = 43.125, 36.375
    lon_min, lon_max = -83.125, -70.000

    # 2-D mask using true lat/lon
    mask = (
        (ds["lat"] >= lat_min) & (ds["lat"] <= lat_max) &
        (ds["lon"] >= lon_min) & (ds["lon"] <= lon_max)
    )

    # keep only pixels inside region (others dropped)
    filtered = ds.where(mask, drop=True)
    return filtered


def main():
    input_base_dir = "/home/ellab/air_pollution/src/data/new_hcho"   # HCHO input
    output_base_dir = "/home/ellab/air_pollution/src/data/clean_hcho"  # HCHO output

    for year in range(2005, 2025):  # inclusive of 2024
        year_dir = os.path.join(input_base_dir, str(year))
        if not os.path.isdir(year_dir):
            continue

        output_year_dir = os.path.join(output_base_dir, str(year))
        os.makedirs(output_year_dir, exist_ok=True)

        for filename in os.listdir(year_dir):
            if filename.endswith(".he5"):
                file_path = os.path.join(year_dir, filename)
                try:
                    ds = extract_data(file_path)
                    filtered_hcho = filter(ds)

                    output_filename = filename.replace(".he5", "_clean.nc")
                    output_path = os.path.join(output_year_dir, output_filename)

                    filtered_hcho.to_netcdf(output_path)
                    print(f"Saved: {output_path}")

                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")


if __name__ == "__main__":
    main()
