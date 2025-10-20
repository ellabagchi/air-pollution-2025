import h5py
import numpy as np
import xarray as xr

# --- HDF4 imports for fallback ---
from pyhdf.SD import SD, SDC

def list_hdf4_datasets(file_path):
    """Print available SDS dataset names in an HDF4 file (useful if the variable name differs)."""
    sd = SD(file_path, SDC.READ)
    for name, idx in sd.datasets().items():
        print(name)
    sd.end()

def extract_data(file_path):
    # Try HDF5 first
    try:
        with h5py.File(file_path, 'r') as f:  # HDF5 path
            # ***change variable below if needed for HDF5 files
            data_path = '/CMGgrid/Data_Fields/AOD_055'
            data = f[data_path][()].copy()

    except (OSError, KeyError):
        # Fall back to HDF4
        sd = SD(file_path, SDC.READ)
        # ***If this raises a KeyError, call list_hdf4_datasets(file_path) to see actual names

        sds = sd.select('AOD_055')
        data = sds[:].astype(float)

        attrs = sds.attributes()
        fill_value = attrs.get('_FillValue', -28672.0)
        scale_factor = attrs.get('scale_factor', 1.0)

        # Clean and scale
        data = np.where(data == fill_value, np.nan, data)
        data = data * scale_factor



    # reconstruct lat/lon grid (unchanged)
    n_lat, n_lon = data.shape
    lat_start = -90 + 0.05 / 2
    lat_end = 90 - 0.05 / 2
    lon_start = -180 + 0.05 / 2
    lon_end = 180 - 0.05 / 2

    latitudes = np.linspace(lat_start, lat_end, n_lat)
    longitudes = np.linspace(lon_start, lon_end, n_lon)

    ds = xr.Dataset(
        {
            'AOD': (['lat', 'lon'], data)  # ***change variable name if needed
        },
        coords={
            'lat': latitudes,
            'lon': longitudes
        }
    )
    return ds

def filter(ds):
    lat_max, lat_min = 43.150, 36.350
    lon_min, lon_max = -83.150, -70.100
    filtered_data = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    return filtered_data

def main():
    # ***change this path to your specific file
    file_path = "/home/ellab/air_pollution/src/data/new_aod/2005/maiac_aod_20050101.hdf"

    # If variable name errors occur with HDF4, uncomment this once to inspect dataset names:
    # list_hdf4_datasets(file_path); return

    ds = extract_data(file_path)
    filtered = filter(ds)

    # ***change output filename if you want
    output_path = "AOD_clean010105.nc"
    filtered.to_netcdf(output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()



