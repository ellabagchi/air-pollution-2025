from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.ensemble import RandomForestRegressor
from shapely.geometry import MultiPoint, Point

from datetime import datetime

# NEW imports for Option A
from pyproj import Transformer
from scipy.spatial import cKDTree


# -------------------------
# User settings
# -------------------------
START_YEAR = 2005
END_YEAR = 2024

# Requested region bounds
LAT_MIN = 36.125
LAT_MAX = 43.125
LON_MIN = -83.125
LON_MAX = -70.000

GRID_STEP_DEG = 0.05  # start coarse, refine later if you want

# Data-quality rule used when creating monthly dataset
MIN_VALID_DAYS_PER_MONTH = 10

# Attribution (edit to what you want)
CREATED_BY = "Ella Bagchi"
PROJECT_NAME = "air_pollution PM2.5 ML interpolation"


# -------------------------
# Helper: add spatial features
# -------------------------
def add_spatial_features(df, transformer, kdtree):
    """
    Adds projected x/y (meters) and distance to nearest training site (km).
    df must contain Latitude and Longitude columns.
    Returns a COPY of df with new columns:
      - x_m
      - y_m
      - dist_nn_km
    """
    df2 = df.copy()

    # Project lon/lat -> x/y meters
    x, y = transformer.transform(
        df2["Longitude"].to_numpy(dtype=float),
        df2["Latitude"].to_numpy(dtype=float),
    )
    df2["x_m"] = x.astype("float32")
    df2["y_m"] = y.astype("float32")

    # Distance to nearest training site (meters -> km)
    dist_m, _ = kdtree.query(np.column_stack([df2["x_m"].values, df2["y_m"].values]), k=1)
    df2["dist_nn_km"] = (dist_m / 1000.0).astype("float32")

    return df2


# -------------------------
# Paths
# -------------------------
splits_dir = Path("/home/ellab/air_pollution/src/data/ml_data/splits_3way")
out_dir = Path("/home/ellab/air_pollution/src/data/ml_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

out_nc = out_dir / (
    f"pm25_pred_rf_{START_YEAR}_{END_YEAR}_"
    f"bbox_{LAT_MIN:.3f}_{LAT_MAX:.3f}_{LON_MIN:.3f}_{LON_MAX:.3f}_"
    f"step{str(GRID_STEP_DEG).replace('.','p')}_optA.nc"
)


# -------------------------
# Load train data and set up spatial feature tools
# -------------------------
train = pd.read_parquet(splits_dir / "train.parquet")

# Projector: lon/lat -> x/y meters
# EPSG:5070 is good for CONUS distances (meters)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

# Build KDTree from UNIQUE training site locations
train_sites = train[["Latitude", "Longitude"]].drop_duplicates().copy()
tx, ty = transformer.transform(
    train_sites["Longitude"].to_numpy(dtype=float),
    train_sites["Latitude"].to_numpy(dtype=float),
)
train_sites["x_m"] = tx.astype("float32")
train_sites["y_m"] = ty.astype("float32")

kdtree = cKDTree(np.column_stack([train_sites["x_m"].values, train_sites["y_m"].values]))


# -------------------------
# Train model using Option A features
# -------------------------
train_feat = add_spatial_features(train, transformer, kdtree)

feature_cols = ["x_m", "y_m", "dist_nn_km", "year", "month"]
X_train = train_feat[feature_cols]
y_train = train_feat["pm25_monthly_mean"]

model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)


# -------------------------
# Build convex hull from training sites ("old way")
# (Hull is still computed from lat/lon — NOT affected by Option A features)
# -------------------------
sites = train[["Latitude", "Longitude"]].drop_duplicates()
site_points = [Point(float(lon), float(lat)) for lat, lon in zip(sites["Latitude"], sites["Longitude"])]
hull = MultiPoint(site_points).convex_hull

print("Hull geom type:", hull.geom_type)


# -------------------------
# Build fixed regional grid (only your bounds)
# -------------------------
lats = np.arange(LAT_MIN, LAT_MAX + GRID_STEP_DEG, GRID_STEP_DEG, dtype=np.float32)
lons = np.arange(LON_MIN, LON_MAX + GRID_STEP_DEG, GRID_STEP_DEG, dtype=np.float32)

lon2d, lat2d = np.meshgrid(lons, lats)  # shape (nlat, nlon)
print("Grid shape (lat, lon):", lon2d.shape)


# -------------------------
# Mask: keep only grid points inside (or on) hull
# Use covers() to include boundary points
# -------------------------
mask = np.zeros(lon2d.shape, dtype=bool)

for i in range(lon2d.shape[0]):
    for j in range(lon2d.shape[1]):
        mask[i, j] = hull.covers(Point(float(lon2d[i, j]), float(lat2d[i, j])))

inside_cells = int(mask.sum())
print("Inside-hull cells:", inside_cells)


# -------------------------
# Time axis (240 months)
# -------------------------
times = []
years = []
months = []

for y in range(START_YEAR, END_YEAR + 1):
    for m in range(1, 13):
        years.append(y)
        months.append(m)
        times.append(np.datetime64(f"{y}-{m:02d}-01"))

times = np.array(times, dtype="datetime64[ns]")
years = np.array(years, dtype=int)
months = np.array(months, dtype=int)


# -------------------------
# Pre-allocate prediction cube (time, lat, lon)
# NaN outside hull
# -------------------------
pred = np.full((len(times), len(lats), len(lons)), np.nan, dtype=np.float32)

inside_lat = lat2d[mask].astype(np.float32)
inside_lon = lon2d[mask].astype(np.float32)

base_grid_df = pd.DataFrame({
    "Latitude": inside_lat,
    "Longitude": inside_lon
})

# Add Option A spatial features to grid ONCE (x_m, y_m, dist_nn_km)
base_grid_feat = add_spatial_features(base_grid_df, transformer, kdtree)


# -------------------------
# Predict for every month
# -------------------------
for t in range(len(times)):
    y = int(years[t])
    m = int(months[t])

    df_feat = base_grid_feat.copy()
    df_feat["year"] = y
    df_feat["month"] = m

    yhat = model.predict(df_feat[feature_cols]).astype(np.float32)

    slice2d = np.full(mask.shape, np.nan, dtype=np.float32)
    slice2d[mask] = yhat
    pred[t, :, :] = slice2d

    if (t + 1) % 12 == 0:
        print(f"Finished year {y}")


# -------------------------
# Build xarray Dataset + METADATA
# -------------------------
creation_date = datetime.now().strftime("%Y-%m-%d")

ds = xr.Dataset(
    data_vars={
        "pm25_pred": (("time", "lat", "lon"), pred)
    },
    coords={
        "time": times,
        "lat": lats,
        "lon": lons,
    }
)

# ---- Variable metadata ----
ds["pm25_pred"].attrs = {
    "long_name": "Predicted monthly mean PM2.5 concentration",
    "standard_name": "mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air",
    "units": "ug m-3",
    "description": (
        "Monthly mean PM2.5 concentration predicted by a Random Forest regression model "
        "trained on monitoring-site monthly averages."
    ),
    "cell_methods": "time: mean (monthly)",
    "valid_range": [0.0, 100.0],
    "missing_value": np.nan,
    "model": "RandomForestRegressor (scikit-learn)",
}

# ---- Coordinate metadata ----
ds["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
ds["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
ds["time"].attrs = {"standard_name": "time"}

# ---- Global metadata ----
ds.attrs = {
    "title": "Machine-learning-based gridded PM2.5 estimates (monthly)",
    "summary": (
        "Monthly PM2.5 concentrations were estimated on a regular latitude–longitude grid "
        "using a Random Forest regression model trained on monitoring-site monthly averages. "
        f"Predictions are provided for each month from {START_YEAR} through {END_YEAR} within a "
        "fixed regional bounding box and masked to the convex hull of training monitoring sites."
    ),
    "source": "Ground-based PM2.5 monitoring data (from the EPA) aggregated from daily to monthly means",
    "methodology": (
        "A RandomForestRegressor model was trained using predictors derived from location and time. "
        "Spatial features include projected coordinates (x/y in meters; EPSG:5070) and distance to nearest "
        "training monitoring site (km), plus year and month. Monitoring locations were split into spatially "
        "disjoint training/validation/test sets. The trained model was applied to a regular grid to produce "
        "gridded monthly PM2.5 fields."
    ),
    "training_data": (
        f"Training split: {splits_dir / 'train.parquet'}; "
        f"monthly means require at least {MIN_VALID_DAYS_PER_MONTH} valid days per site-month."
    ),
    "validation_strategy": "Spatial site-based split (70% train, 15% val, 15% test by monitoring site)",
    "note_temporal_split": "No temporal holdout used (time periods are represented in training data).",
    "spatial_domain": (
        f"Lat {LAT_MIN} to {LAT_MAX} (degrees_north), "
        f"Lon {LON_MIN} to {LON_MAX} (degrees_east; negative is west)"
    ),
    "time_coverage_start": f"{START_YEAR}-01-01",
    "time_coverage_end": f"{END_YEAR}-12-31",
    "grid_resolution_degrees": float(GRID_STEP_DEG),
    "mask": "NaN outside convex hull polygon computed from training monitoring sites",
    "created_by": CREATED_BY,
    "project": PROJECT_NAME,
    "creation_date": creation_date,
}


# -------------------------
# Write NetCDF with compression (netCDF4 backend)
# -------------------------
encoding = {
    "pm25_pred": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": np.float32(np.nan),
    }
}

ds.to_netcdf(out_nc, engine="netcdf4", encoding=encoding)

print("Saved NetCDF:", out_nc)
print("Done.")
