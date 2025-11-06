import os
import numpy as np
import xarray as xr
from pyhdf.SD import SD, SDC

def extract_data(file_path):
    """
    Reconstruct AOD from MAIAC compact format for one file:
    - read Compact_AOD_055 + indexing arrays
    - average all nAOD records per cell
    - apply fill/scale from AOD_055 SDS
    - return global gridded Dataset with AOD_055_compact(lat, lon)
    """
    sd = SD(file_path, SDC.READ)

    # --- read compact arrays ---
    compact_aod = sd.select("Compact_AOD_055")[:].astype(np.float32)[0]
    line_arr    = sd.select("Line")[:].astype(np.int32)[0]
    sample_arr  = sd.select("Sample")[:].astype(np.int32)[0]
    offset_arr  = sd.select("Offset_AOD_055")[:].astype(np.int64)[0]
    nAOD_arr    = sd.select("nAOD")[:].astype(np.int32)[0]

    # --- get scaling / fill info from AOD_055 SDS ---
    sds_grid = sd.select("AOD_055")
    attrs    = sds_grid.attributes()
    fill_value   = attrs.get("_FillValue", -28672)
    valid_range  = attrs.get("valid_range", [0, 6000])
    scale_factor = attrs.get("scale_factor", 0.001)

    valid_min = float(valid_range[0])
    valid_max = float(valid_range[1])

    # --- grid dimensions ---
    nlines  = 3600
    nsamples = 7200
    AODImg  = np.full((nlines, nsamples), np.nan, dtype=np.float32)

    # --- reconstruct: mean over all nAOD records per cell ---
    ncells = line_arr.shape[0]
    for i in range(ncells):
        line   = int(line_arr[i])
        sample = int(sample_arr[i])
        offset = int(offset_arr[i])
        nAOD   = int(nAOD_arr[i])

        if nAOD <= 0:
            continue
        if not (0 <= line < nlines and 0 <= sample < nsamples):
            continue

        start = offset
        end   = offset + nAOD
        if start < 0 or end > compact_aod.size:
            continue

        vals = compact_aod[start:end].astype(np.float64)
        if vals.size == 0:
            continue

        AODImg[line, sample] = np.mean(vals)

    # --- clean & scale ---
    AOD_raw = AODImg.copy()
    invalid_mask = (
        (AOD_raw == fill_value) |
        (AOD_raw < valid_min) |
        (AOD_raw > valid_max)
    )
    AOD_raw[invalid_mask] = np.nan
    AOD_phys = AOD_raw * scale_factor

    # --- build lat/lon axes (0.05° global grid) ---
    line_indices   = np.arange(nlines)
    sample_indices = np.arange(nsamples)
    latitudes  = 90.0  - line_indices * 0.05      # 90 -> -90 (DESCENDING)
    longitudes = -180.0 + sample_indices * 0.05   # -180 -> 180 (ASCENDING)

    sd.end()

    # --- wrap in xarray dataset (global) ---
    ds = xr.Dataset(
        {"AOD_055_compact": (["lat", "lon"], AOD_phys.astype(np.float32))},
        coords={"lat": latitudes, "lon": longitudes},
    )
    ds["AOD_055_compact"].attrs.update({
        "long_name": "AOD at 550 nm from MAIAC compact product (mean over all overpasses)",
        "units": "1",
        "description": "Reconstructed from Compact_AOD_055 using mean of all nAOD records per CMG grid cell.",
    })
    ds["lat"].attrs.update({"long_name": "latitude", "units": "degrees_north"})
    ds["lon"].attrs.update({"long_name": "longitude", "units": "degrees_east"})
    
    return ds

def filter_region(ds):
    """Subset to region of interest (gridded)."""
    lat_max, lat_min = 43.150, 36.350
    lon_min, lon_max = -83.150, -70.100

    # lat is descending, so slice(lat_max, lat_min)
    return ds.sel(
        lat=slice(lat_max, lat_min),
        lon=slice(lon_min, lon_max)
    )

def add_points_from_region(ds_reg):
    """
    From a regional gridded Dataset (lat, lon, AOD_055_compact),
    build point-style arrays and return a new Dataset with both.
    """
    A = ds_reg["AOD_055_compact"].values  # 2D (lat, lon)
    lat_reg = ds_reg["lat"].values
    lon_reg = ds_reg["lon"].values

    # build 2D lat/lon grids for indexing
    LatGrid, LonGrid = np.meshgrid(lat_reg, lon_reg, indexing="ij")

    valid = np.isfinite(A)
    aod_pts = A[valid]
    lat_pts = LatGrid[valid]
    lon_pts = LonGrid[valid]

    ds_out = xr.Dataset(
        {
            "AOD_055_compact_gridded": (("lat", "lon"), A.astype(np.float32)),
            "AOD_055_compact_points": (("points",), aod_pts.astype(np.float32)),
            "lat_points": (("points",), lat_pts.astype(np.float32)),
            "lon_points": (("points",), lon_pts.astype(np.float32)),
        },
        coords={
            "lat": lat_reg,
            "lon": lon_reg,
        },
    )

    # copy attrs
    ds_out["AOD_055_compact_gridded"].attrs.update(
        ds_reg["AOD_055_compact"].attrs
    )
    ds_out["AOD_055_compact_gridded"].attrs["description"] += " Subset to specified region."
    ds_out["AOD_055_compact_points"].attrs.update({
        "long_name": "AOD at 550 nm (compact MAIAC, regional points)",
        "units": "1",
        "description": (
            "Point sample of compact AOD within specified lat/lon bounds, "
            "averaged per CMG grid cell and scaled as in gridded field."
        ),
    })
    ds_out["lat"].attrs.update({"long_name": "latitude", "units": "degrees_north"})
    ds_out["lon"].attrs.update({"long_name": "longitude", "units": "degrees_east"})
    ds_out["lat_points"].attrs.update({"long_name": "latitude of points", "units": "degrees_north"})
    ds_out["lon_points"].attrs.update({"long_name": "longitude of points", "units": "degrees_east"})

    return ds_out

def main():
    # --- paths ---
    input_base_dir  = "/home/ellab/air_pollution/src/data/new_aod"
    output_base_dir = "/home/ellab/air_pollution/src/data/clean_aod"

    # --- years to process ---
    START_YEAR = 2007
    END_YEAR   = 2007

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
                ds_global = extract_data(file_path)      # global compact AOD
                ds_reg    = filter_region(ds_global)     # regional gridded
                print(year, filename, float(ds_reg["AOD_055_compact"].count())) ############ DELETE LATER
                ds_out    = add_points_from_region(ds_reg)  # add points

                output_filename = filename.replace(".hdf", "_clean.nc")
                output_path = os.path.join(out_year_dir, output_filename)

                # compress all data variables
                encoding = {
                    var: {"zlib": True, "complevel": 4}
                    for var in ds_out.data_vars
                }

                ds_out.to_netcdf(output_path, encoding=encoding)
                print(f"Saved: {output_path}")

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    main()
