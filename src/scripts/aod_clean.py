
# DO NOT USE
import os
import numpy as np
import xarray as xr
from pyhdf.SD import SD, SDC

def extract_data(file_path):
    """Extract AOD_055 from HDF4 MAIAC CMG file, apply fill/scale, and reconstruct 0.05° grid."""
    # --- Read AOD_055 from HDF4 ---
    sd = SD(file_path, SDC.READ)
    sds = sd.select('AOD_055')  # if error, run list_hdf4_datasets(file_path) to see true names
    data = sds[:].astype(float)

    # --- Attributes ---
    attrs = sds.attributes()
    fill_value = attrs.get('_FillValue', -28672.0)
    scale_factor = attrs.get('scale_factor', 1.0)

    # --- Clean & scale ---
    data = np.where(data == fill_value, np.nan, data)
    data = data * scale_factor

    # --- Build 0.05° global lat/lon grid ---
    n_lat, n_lon = data.shape
    res = 0.05
    latitudes = np.linspace(-90 + res / 2, 90 - res / 2, n_lat)   # this is where you extract actual lat/lon
    longitudes = np.linspace(-180 + res / 2, 180 - res / 2, n_lon)

    sd.end()

    # --- Wrap in xarray dataset ---
    ds = xr.Dataset(
        {"AOD": (["lat", "lon"], data)},
        coords={"lat": latitudes, "lon": longitudes}
    )
    return ds

def filter_region(ds):
    """Subset to region of interest."""
    lat_max, lat_min = 43.150, 36.350
    lon_min, lon_max = -83.150, -70.100
    return ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

def main():
    # --- paths ---
    input_base_dir  = "/home/ellab/air_pollution/src/data/new_aod"
    output_base_dir = "/home/ellab/air_pollution/src/data/clean_aod"

    # --- years to process ---
    START_YEAR = 2006
    END_YEAR   = 2016  # change to 2024, etc.

    for year in range(START_YEAR, END_YEAR + 1):
        year_dir = os.path.join(input_base_dir, str(year))
        if not os.path.isdir(year_dir):
            print(f"Skipping {year} (no folder).")
            continue

        out_year_dir = os.path.join(output_base_dir, str(year))
        os.makedirs(out_year_dir, exist_ok=True)

        for filename in sorted(os.listdir(year_dir)):
            if not filename.lower().endswith(".hdf"):
                continue
            file_path = os.path.join(year_dir, filename)
            try:
                ds = extract_data(file_path)
                filtered = filter_region(ds)

                output_filename = filename.replace(".hdf", "_clean.nc")
                output_path = os.path.join(out_year_dir, output_filename)

                filtered.to_netcdf(output_path)
                print(f"Saved: {output_path}")

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    main()



# from pyhdf.SD import SD, SDC
# hdf_file = SD('/home/ellab/air_pollution/src/data/new_aod/2008/maiac_aod_20080120.hdf', SDC.READ)

# data_path = '/CMGgrid/Data_Fields/AOD_055' 

# print(hdf_file.datasets)
# # Select a dataset by name (not a full path!)
# dataset = hdf_file.select('AOD_055')  # dataset name only, not /CMGgrid/Data_Fields/

# # Read data into a NumPy array
# data = dataset.get()

# print(data.shape)
# print(data)





# import h5py  # left in place even if unused
# import pyhdf
# import numpy as np
# import xarray as xr
# import os

# from pyhdf.SD import SD, SDC

# def extract_data(file_path):
#     # open HDF4 with pyhdf
#     hdf = SD(file_path, SDC.READ)

#     # ***change variable below
#     data_path = '/CMGgrid/Data_Fields/AOD_055'
#     # data_path = '/HDFEOS/GRIDS/ColumnAmountO3/Data Fields/ColumnAmountO3'  # for O3
#     # data_path = '/HDFEOS/GRIDS/OMI Total Column Amount HCHO/Data Fields/ColumnAmountHCHO'  # for HCHO

#     # In HDF4 (pyhdf), select by dataset name only (last path component)
#     dataset_name = data_path.split('/')[-1]
#     sds = hdf.select(dataset_name)
#     data = sds.get()  # numpy array, typically int16

#     # ----- NEW: read attributes & apply mask/scale -----
#     attrs = sds.attributes()

#     # Common attribute names across NASA/HDF products
#     fill = (
#         attrs.get('_FillValue', None)
#         or attrs.get('fill_value', None)
#         or attrs.get('FillValue', None)
#     )
#     valid_range = attrs.get('valid_range', None)

#     # Scale/offset can appear under different names
#     scale = attrs.get('scale_factor', None)
#     if scale is None:
#         scale = attrs.get('Slope', None)
#     offset = attrs.get('add_offset', None)
#     if offset is None:
#         offset = attrs.get('Intercept', None)

#     data = np.array(data)  # ensure ndarray
#     data = data.astype(np.float32)

#     # Mask fill values
#     if fill is not None:
#         data[data == np.float32(fill)] = np.nan

#     # Mask outside valid_range if present
#     if valid_range is not None and len(valid_range) == 2:
#         vmin, vmax = float(valid_range[0]), float(valid_range[1])
#         bad = (data < vmin) | (data > vmax)
#         data[bad] = np.nan

#     # Apply scale/offset
#     if scale is not None:
#         data = data * np.float32(scale)
#     if offset is not None:
#         data = data + np.float32(offset)

#     # Close HDF handles
#     sds.endaccess()
#     hdf.end()
#     # ----- END NEW -----

#     # Reconstruct lat/lon grid (your original approach)
#     n_lat, n_lon = data.shape
#     lat_start = -90 + 0.05 / 2
#     lat_end = 90 - 0.05 / 2
#     lon_start = -180 + 0.05 / 2
#     lon_end = 180 - 0.05 / 2  # fixed small typo in your original

#     latitudes = np.linspace(lat_start, lat_end, n_lat)
#     longitudes = np.linspace(lon_start, lon_end, n_lon)

#     # If your product stores latitude descending, you may need:
#     # if latitudes[0] > latitudes[-1]:
#     #     latitudes = latitudes[::-1]
#     #     data = data[::-1, :]

#     ds = xr.Dataset(
#         {'AOD': (['lat', 'lon'], data.astype(np.float32))},  # ensure float32 output
#         coords={'lat': latitudes, 'lon': longitudes}
#     )

#     return ds

# def filter(ds):
#     lat_max, lat_min = 43.150, 36.300
#     lon_min, lon_max = -83.150, -70.000
#     filtered_data = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
#     return filtered_data

# def main():
#     # ***change variables below
#     input_base_dir = "/home/ellab/air_pollution/src/data/new_aod"
#     # input_base_dir = "/home/ellab/air_pollution/src/data/new_o3"  # O3
#     # input_base_dir = "/home/ellab/air_pollution/src/data/new_hcho"  # HCHO

#     output_base_dir = "/home/ellab/air_pollution/src/data/clean_aod"
#     # output_base_dir = "/home/ellab/air_pollution/src/data/clean_o3"  # O3
#     # output_base_dir = "/home/ellab/air_pollution/src/data/clean_hcho"  # HCHO

#     for year in range(2005, 2006):
#         year_dir = os.path.join(input_base_dir, str(year))
#         if not os.path.isdir(year_dir):
#             continue

#         output_year_dir = os.path.join(output_base_dir, str(year))
#         os.makedirs(output_year_dir, exist_ok=True)

#         for filename in os.listdir(year_dir):
#             if filename.endswith(".hdf"):
#                 file_path = os.path.join(year_dir, filename)
#                 try:
#                     ds = extract_data(file_path)
#                     filtered = filter(ds)

#                     output_filename = filename.replace(".hdf", "_clean.nc")
#                     output_path = os.path.join(output_year_dir, output_filename)
#                     filtered.to_netcdf(output_path)
#                     print(f"Saved: {output_path}")
#                 except Exception as e:
#                     print(f"Failed to process {file_path}: {e}")

# if __name__ == "__main__":
#     main()
