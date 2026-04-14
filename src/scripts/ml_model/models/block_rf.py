from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import MultiPoint, Point

from pyproj import Transformer
from scipy.spatial import cKDTree

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# USER SETTINGS
# ============================================================
START_YEAR = 2005
END_YEAR = 2024

# Requested region bounds (NetCDF output grid bounds)
LAT_MIN = 36.125
LAT_MAX = 43.125
LON_MIN = -83.125
LON_MAX = -70.000

GRID_STEP_DEG = 0.05

# Data-quality rule used when creating monthly dataset (if n_days exists)
MIN_VALID_DAYS_PER_MONTH = 10

# Spatial blocking settings
BLOCK_SIZE_KM = 75         # try 50–150 km; larger = harder
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

# Attribution (edit as you like)
CREATED_BY = "Ella Bagchi"
PROJECT_NAME = "air_pollution PM2.5 ML interpolation"

# Path to your full monthly dataset (ALL sites, ALL months)
MONTHLY_PATH = Path("/home/ellab/air_pollution/src/data/pm/pm25_monthly_2005_2024.parquet")

# Output folder
OUT_DIR = Path("/home/ellab/air_pollution/src/data/ml_outputs")


# ============================================================
# HELPERS
# ============================================================
def report_metrics(split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{split_name:5s}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")


def add_spatial_features(df: pd.DataFrame, transformer: Transformer, kdtree: cKDTree) -> pd.DataFrame:
    """
    Adds projected x/y (meters; EPSG:5070) and distance to nearest TRAIN site (km).
    Requires df has Latitude/Longitude.
    Returns a COPY with columns: x_m, y_m, dist_nn_km.
    """
    df2 = df.copy()

    x, y = transformer.transform(
        df2["Longitude"].to_numpy(dtype=float),
        df2["Latitude"].to_numpy(dtype=float),
    )
    df2["x_m"] = x.astype("float32")
    df2["y_m"] = y.astype("float32")

    dist_m, _ = kdtree.query(np.column_stack([df2["x_m"].values, df2["y_m"].values]), k=1)
    df2["dist_nn_km"] = (dist_m / 1000.0).astype("float32")

    return df2


def make_spatial_blocks_split(
    df_monthly: pd.DataFrame,
    transformer: Transformer,
    block_size_km: float,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    """
    Assigns a split label to each SITE using spatial blocks (grid in projected meters).
    Returns df_monthly with a new column 'split' in {'train','val','test'}.
    """

    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0")

    # Unique sites
    sites = df_monthly[["Latitude", "Longitude"]].drop_duplicates().copy()

    # Project to meters
    x, y = transformer.transform(
        sites["Longitude"].to_numpy(dtype=float),
        sites["Latitude"].to_numpy(dtype=float),
    )
    sites["x_m"] = x.astype("float32")
    sites["y_m"] = y.astype("float32")

    # Block ids
    block_size_m = float(block_size_km) * 1000.0
    bx = np.floor(sites["x_m"] / block_size_m).astype(int)
    by = np.floor(sites["y_m"] / block_size_m).astype(int)
    sites["block_id"] = bx.astype(str) + "_" + by.astype(str)

    # Assign blocks to splits (reproducible)
    blocks = sites["block_id"].drop_duplicates().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)

    n_blocks = len(blocks)
    n_train = int(round(train_frac * n_blocks))
    n_val = int(round(val_frac * n_blocks))
    # remainder => test

    train_blocks = set(blocks[:n_train])
    val_blocks = set(blocks[n_train:n_train + n_val])
    test_blocks = set(blocks[n_train + n_val:])

    def block_to_split(b: str) -> str:
        if b in train_blocks:
            return "train"
        if b in val_blocks:
            return "val"
        if b in test_blocks:
            return "test"
        return "test"

    sites["split"] = sites["block_id"].map(block_to_split)

    # Merge back to monthly rows
    df_out = df_monthly.merge(
        sites[["Latitude", "Longitude", "split"]],
        on=["Latitude", "Longitude"],
        how="left",
    )

    if df_out["split"].isna().any():
        raise ValueError("Some rows did not get a split assignment. Check Latitude/Longitude matching.")

    return df_out


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_nc = OUT_DIR / (
        f"pm25_pred_rf_blocked_{START_YEAR}_{END_YEAR}_"
        f"bbox_{LAT_MIN:.3f}_{LAT_MAX:.3f}_{LON_MIN:.3f}_{LON_MAX:.3f}_"
        f"step{str(GRID_STEP_DEG).replace('.','p')}_"
        f"block{BLOCK_SIZE_KM}km.nc"
    )

    print("Reading monthly data:", MONTHLY_PATH)
    df = pd.read_parquet(MONTHLY_PATH)

    # Restrict years if needed
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    # Re-apply n_days rule if column exists
    if "n_days" in df.columns:
        before = len(df)
        df = df[df["n_days"] >= MIN_VALID_DAYS_PER_MONTH].copy()
        after = len(df)
        print(f"Applied n_days >= {MIN_VALID_DAYS_PER_MONTH}: kept {after:,}/{before:,} rows")

    # Projection
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

    # Spatial-blocked splits
    df = make_spatial_blocks_split(
        df_monthly=df,
        transformer=transformer,
        block_size_km=BLOCK_SIZE_KM,
        seed=RANDOM_SEED,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
    )

    # Split summaries
    site_split = df[["Latitude", "Longitude", "split"]].drop_duplicates()
    print("\nSites by split:")
    print(site_split["split"].value_counts().to_string())

    print("\nRows by split:")
    print(df["split"].value_counts().to_string())

    # Build TRAIN-only KDTree
    train_sites = site_split[site_split["split"] == "train"][["Latitude", "Longitude"]].drop_duplicates().copy()

    tx, ty = transformer.transform(
        train_sites["Longitude"].to_numpy(dtype=float),
        train_sites["Latitude"].to_numpy(dtype=float),
    )
    train_sites["x_m"] = tx.astype("float32")
    train_sites["y_m"] = ty.astype("float32")

    kdtree = cKDTree(np.column_stack([train_sites["x_m"].values, train_sites["y_m"].values]))

    # Train model on TRAIN rows
    train_rows = df[df["split"] == "train"].copy()
    train_feat = add_spatial_features(train_rows, transformer, kdtree)

    feature_cols = ["x_m", "y_m", "dist_nn_km", "year", "month"]
    X_train = train_feat[feature_cols]
    y_train = train_feat["pm25_monthly_mean"]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print("\nFitting Random Forest...")
    model.fit(X_train, y_train)

    # Evaluate on VAL and TEST rows
    print("\nBlocked-split evaluation:")
    for split_name in ["val", "test"]:
        df_split = df[df["split"] == split_name].copy()
        df_split_feat = add_spatial_features(df_split, transformer, kdtree)

        X = df_split_feat[feature_cols]
        y_true = df_split_feat["pm25_monthly_mean"].to_numpy()
        y_pred = model.predict(X)

        report_metrics(split_name.upper(), y_true, y_pred)

    # Convex hull from TRAIN sites only (for masking)
    train_site_points = [
        Point(float(lon), float(lat))
        for lat, lon in zip(train_sites["Latitude"], train_sites["Longitude"])
    ]
    hull = MultiPoint(train_site_points).convex_hull
    print("\nHull geom type:", hull.geom_type)

    # Build output grid (bbox)
    lats = np.arange(LAT_MIN, LAT_MAX + GRID_STEP_DEG, GRID_STEP_DEG, dtype=np.float32)
    lons = np.arange(LON_MIN, LON_MAX + GRID_STEP_DEG, GRID_STEP_DEG, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lons, lats)
    print("Grid shape (lat, lon):", lon2d.shape)

    # Mask inside hull (covers => includes boundary)
    mask = np.zeros(lon2d.shape, dtype=bool)
    for i in range(lon2d.shape[0]):
        for j in range(lon2d.shape[1]):
            mask[i, j] = hull.covers(Point(float(lon2d[i, j]), float(lat2d[i, j])))

    inside_cells = int(mask.sum())
    print("Inside-hull cells:", inside_cells)

    # Time axis (240 months)
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

    # Pre-allocate predictions (time, lat, lon)
    pred = np.full((len(times), len(lats), len(lons)), np.nan, dtype=np.float32)

    # Base grid points inside hull
    inside_lat = lat2d[mask].astype(np.float32)
    inside_lon = lon2d[mask].astype(np.float32)
    base_grid_df = pd.DataFrame({"Latitude": inside_lat, "Longitude": inside_lon})
    base_grid_feat = add_spatial_features(base_grid_df, transformer, kdtree)

    # Predict each month
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

    # Build xarray dataset + metadata
    creation_date = datetime.now().strftime("%Y-%m-%d")

    ds = xr.Dataset(
        data_vars={"pm25_pred": (("time", "lat", "lon"), pred)},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    ds["pm25_pred"].attrs = {
        "long_name": "Predicted monthly mean PM2.5 concentration",
        "standard_name": "mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air",
        "units": "ug m-3",
        "description": (
            "Monthly mean PM2.5 predicted by a Random Forest regression model with spatial features "
            "(projected x/y and distance to nearest training site) and spatial-blocked train/val/test split."
        ),
        "cell_methods": "time: mean (monthly)",
        "valid_range": [0.0, 100.0],
        "missing_value": np.nan,
        "model": "RandomForestRegressor (scikit-learn)",
    }

    ds["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    ds["time"].attrs = {"standard_name": "time"}

    ds.attrs = {
        "title": "Machine-learning-based gridded PM2.5 estimates (monthly) - spatially blocked model",
        "summary": (
            "Monthly PM2.5 concentrations were estimated on a regular latitude–longitude grid using a "
            "Random Forest regression model trained on monitoring-site monthly averages. Spatial features "
            "include projected coordinates (EPSG:5070) and distance to nearest training monitoring site (km), "
            "plus year and month. Train/validation/test splits were created using spatial blocking to better "
            "test generalization to new regions. Predictions are masked to the convex hull of training sites."
        ),
        "source": "Ground-based PM2.5 monitoring data (EPA) aggregated from daily to monthly means",
        "methodology": (
            "RandomForestRegressor trained with predictors: x_m, y_m, dist_nn_km, year, month. "
            f"Spatial blocking used blocks of {BLOCK_SIZE_KM} km in EPSG:5070 projected space."
        ),
        "training_data": str(MONTHLY_PATH),
        "validation_strategy": f"Spatial blocking by {BLOCK_SIZE_KM} km blocks; blocks assigned to train/val/test.",
        "note_temporal_split": "No temporal holdout used (time periods are represented in training data).",
        "spatial_domain": (
            f"Lat {LAT_MIN} to {LAT_MAX} (degrees_north), "
            f"Lon {LON_MIN} to {LON_MAX} (degrees_east; negative is west)"
        ),
        "time_coverage_start": f"{START_YEAR}-01-01",
        "time_coverage_end": f"{END_YEAR}-12-31",
        "grid_resolution_degrees": float(GRID_STEP_DEG),
        "mask": "NaN outside convex hull polygon computed from training monitoring sites",
        "min_valid_days_per_month_rule": str(MIN_VALID_DAYS_PER_MONTH),
        "created_by": CREATED_BY,
        "project": PROJECT_NAME,
        "creation_date": creation_date,
    }

    # Write NetCDF with compression (netCDF4 backend)
    encoding = {
        "pm25_pred": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }

    ds.to_netcdf(out_nc, engine="netcdf4", encoding=encoding)

    print("\nSaved NetCDF:", out_nc)
    print("Done.")


if __name__ == "__main__":
    main()
