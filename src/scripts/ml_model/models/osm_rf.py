from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import MultiPoint, Point, box
from shapely.ops import unary_union

from pyproj import Transformer
from scipy.spatial import cKDTree

import geopandas as gpd
import osmnx as ox

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# USER SETTINGS
# ============================================================
START_YEAR = 2005
END_YEAR = 2024

# NYC
LAT_MIN = 40.50
LAT_MAX = 40.95
LON_MIN = -74.30
LON_MAX = -73.65

GRID_STEP_DEG = 0.05

MIN_VALID_DAYS_PER_MONTH = 10

BLOCK_SIZE_KM = 75
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

CREATED_BY = "Ella Bagchi"
PROJECT_NAME = "air_pollution PM2.5 ML interpolation"

MONTHLY_PATH = Path("/home/ellab/air_pollution/src/data/pm/pm25_monthly_2005_2024.parquet")

OUT_DIR = Path("/home/ellab/air_pollution/src/data/ml_outputs")
CACHE_DIR = Path("/home/ellab/air_pollution/src/data/osm_cache")

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROADS_EDGES_PARQUET = CACHE_DIR / (
    f"osm_roads_edges_bbox_{LAT_MIN}_{LAT_MAX}_{LON_MIN}_{LON_MAX}.parquet"
)
SITE_ROAD_FEATURES_PARQUET = CACHE_DIR / (
    f"site_road_distance_features_bbox_{LAT_MIN}_{LAT_MAX}_{LON_MIN}_{LON_MAX}.parquet"
)


# ============================================================
# METRICS
# ============================================================
def report_metrics(split_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"{split_name:5s}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")


# ============================================================
# SPATIAL FEATURES
# ============================================================
def add_spatial_features(
    df: pd.DataFrame,
    transformer: Transformer,
    kdtree: cKDTree
) -> pd.DataFrame:
    df2 = df.copy()

    x, y = transformer.transform(
        df2["Longitude"].to_numpy(dtype=float),
        df2["Latitude"].to_numpy(dtype=float),
    )
    df2["x_m"] = x.astype("float32")
    df2["y_m"] = y.astype("float32")

    dist_m, _ = kdtree.query(
        np.column_stack([df2["x_m"].values, df2["y_m"].values]),
        k=1
    )
    df2["dist_nn_km"] = (dist_m / 1000.0).astype("float32")

    return df2


# ============================================================
# ROAD FEATURES
# ============================================================
def make_points_gdf(latlon_df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        latlon_df.copy(),
        geometry=gpd.points_from_xy(latlon_df["Longitude"], latlon_df["Latitude"]),
        crs="EPSG:4326",
    )


def has_kind(v, allowed: list[str]) -> bool:
    """
    OSM 'highway' can be a string or a list.
    """
    if isinstance(v, list):
        return any(str(x) in allowed for x in v)
    if pd.isna(v):
        return False
    return str(v) in allowed


def load_or_download_roads_edges() -> gpd.GeoDataFrame:
    """
    Download drivable OSM roads once and cache them.
    Cache only boolean road-type columns + geometry so Parquet stays simple.
    """
    if ROADS_EDGES_PARQUET.exists():
        print(f"Loading cached OSM roads: {ROADS_EDGES_PARQUET}")
        edges = gpd.read_parquet(ROADS_EDGES_PARQUET)
        if edges.crs is None:
            edges = edges.set_crs("EPSG:4326")
        return edges

    print("Downloading OSM roads (first run only)...")

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(CACHE_DIR)
    ox.settings.log_console = False

    poly = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    G = ox.graph_from_polygon(poly, network_type="drive")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    edges = edges[["highway", "geometry"]].copy()
    edges = edges.set_crs("EPSG:4326")

    edges["is_motorway"] = edges["highway"].apply(
        lambda v: has_kind(v, ["motorway", "motorway_link"])
    )
    edges["is_trunk"] = edges["highway"].apply(
        lambda v: has_kind(v, ["trunk", "trunk_link"])
    )
    edges["is_primary"] = edges["highway"].apply(
        lambda v: has_kind(v, ["primary", "primary_link"])
    )

    edges = edges[["is_motorway", "is_trunk", "is_primary", "geometry"]].copy()

    print(f"Saving cached roads to: {ROADS_EDGES_PARQUET}")
    edges.to_parquet(ROADS_EDGES_PARQUET, index=False)
    return edges


def build_union_geometry(lines_gdf: gpd.GeoDataFrame):
    """
    Project lines to meters and unary-union them into one geometry.
    """
    if lines_gdf is None or len(lines_gdf) == 0:
        return None

    lines_m = lines_gdf.to_crs("EPSG:5070")
    return unary_union(lines_m.geometry.tolist())


def dist_to_union_km(points_gdf: gpd.GeoDataFrame, union_geom) -> np.ndarray:
    """
    Distance from each point to a single merged road geometry, in km.
    """
    if union_geom is None:
        return np.full(len(points_gdf), np.nan, dtype="float32")

    pts_m = points_gdf.to_crs("EPSG:5070")
    d_km = pts_m.geometry.distance(union_geom) / 1000.0
    return d_km.to_numpy(dtype="float32")


def compute_or_load_site_road_features(all_sites_latlon: pd.DataFrame) -> pd.DataFrame:
    """
    Compute road-distance features once for all unique monitoring sites.
    """
    if SITE_ROAD_FEATURES_PARQUET.exists():
        print(f"Loading cached site road features: {SITE_ROAD_FEATURES_PARQUET}")
        return pd.read_parquet(SITE_ROAD_FEATURES_PARQUET)

    edges = load_or_download_roads_edges()

    motorway = edges[edges["is_motorway"]].copy()
    trunk = edges[edges["is_trunk"]].copy()
    primary = edges[edges["is_primary"]].copy()

    motorway_union = build_union_geometry(motorway)
    trunk_union = build_union_geometry(trunk)
    primary_union = build_union_geometry(primary)

    pts_gdf = make_points_gdf(all_sites_latlon)

    print("Computing distance-to-road features for sites...")
    out = all_sites_latlon.copy()
    out["dist_motorway_km"] = dist_to_union_km(pts_gdf, motorway_union)
    out["dist_trunk_km"] = dist_to_union_km(pts_gdf, trunk_union)
    out["dist_primary_km"] = dist_to_union_km(pts_gdf, primary_union)

    print(f"Saving site road features to: {SITE_ROAD_FEATURES_PARQUET}")
    out.to_parquet(SITE_ROAD_FEATURES_PARQUET, index=False)
    return out


# ============================================================
# SPATIAL BLOCKING SPLIT
# ============================================================
def make_spatial_blocks_split(
    df_monthly: pd.DataFrame,
    transformer: Transformer,
    block_size_km: float,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> pd.DataFrame:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0")

    sites = df_monthly[["Latitude", "Longitude"]].drop_duplicates().copy()

    x, y = transformer.transform(
        sites["Longitude"].to_numpy(dtype=float),
        sites["Latitude"].to_numpy(dtype=float),
    )
    sites["x_m"] = x.astype("float32")
    sites["y_m"] = y.astype("float32")

    block_size_m = float(block_size_km) * 1000.0
    bx = np.floor(sites["x_m"] / block_size_m).astype(int)
    by = np.floor(sites["y_m"] / block_size_m).astype(int)
    sites["block_id"] = bx.astype(str) + "_" + by.astype(str)

    blocks = sites["block_id"].drop_duplicates().to_list()
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)

    n_blocks = len(blocks)
    n_train = int(round(train_frac * n_blocks))
    n_val = int(round(val_frac * n_blocks))

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

    df_out = df_monthly.merge(
        sites[["Latitude", "Longitude", "split"]],
        on=["Latitude", "Longitude"],
        how="left",
    )

    if df_out["split"].isna().any():
        raise ValueError("Some rows did not get a split assignment.")

    return df_out


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    out_nc = OUT_DIR / (
        f"NYCpm25_pred_rf_blocked_roads_{START_YEAR}_{END_YEAR}_"
        f"bbox_{LAT_MIN:.3f}_{LAT_MAX:.3f}_{LON_MIN:.3f}_{LON_MAX:.3f}_"
        f"step{str(GRID_STEP_DEG).replace('.','p')}_"
        f"block{BLOCK_SIZE_KM}km.nc"
    )

    print("Reading monthly data:", MONTHLY_PATH)
    df = pd.read_parquet(MONTHLY_PATH)
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)].copy()

    if "n_days" in df.columns:
        before = len(df)
        df = df[df["n_days"] >= MIN_VALID_DAYS_PER_MONTH].copy()
        after = len(df)
        print(f"Applied n_days >= {MIN_VALID_DAYS_PER_MONTH}: kept {after:,}/{before:,} rows")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

    # Add road features to all unique sites, then merge
    all_sites = df[["Latitude", "Longitude"]].drop_duplicates().copy()
    road_feats = compute_or_load_site_road_features(all_sites)
    df = df.merge(road_feats, on=["Latitude", "Longitude"], how="left")

    for c in ["dist_motorway_km", "dist_trunk_km", "dist_primary_km"]:
        df[c] = df[c].astype("float32").fillna(np.float32(999.0))

    # Spatial blocked split
    df = make_spatial_blocks_split(
        df_monthly=df,
        transformer=transformer,
        block_size_km=BLOCK_SIZE_KM,
        seed=RANDOM_SEED,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
    )

    site_split = df[["Latitude", "Longitude", "split"]].drop_duplicates()
    print("\nSites by split:")
    print(site_split["split"].value_counts().to_string())

    print("\nRows by split:")
    print(df["split"].value_counts().to_string())

    # Train-only KDTree for dist_nn_km
    train_sites = site_split[
        site_split["split"] == "train"
    ][["Latitude", "Longitude"]].drop_duplicates().copy()

    tx, ty = transformer.transform(
        train_sites["Longitude"].to_numpy(dtype=float),
        train_sites["Latitude"].to_numpy(dtype=float),
    )
    train_sites["x_m"] = tx.astype("float32")
    train_sites["y_m"] = ty.astype("float32")
    kdtree = cKDTree(np.column_stack([train_sites["x_m"].values, train_sites["y_m"].values]))

    # Train model
    train_rows = df[df["split"] == "train"].copy()
    train_feat = add_spatial_features(train_rows, transformer, kdtree)

    feature_cols = [
        "x_m", "y_m", "dist_nn_km",
        "dist_motorway_km", "dist_trunk_km", "dist_primary_km",
        "year", "month"
    ]

    X_train = train_feat[feature_cols]
    y_train = train_feat["pm25_monthly_mean"]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print("\nFitting Random Forest (blocked + roads)...")
    model.fit(X_train, y_train)

    print("\nBlocked-split evaluation (with roads):")
    for split_name in ["val", "test"]:
        df_split = df[df["split"] == split_name].copy()
        df_split_feat = add_spatial_features(df_split, transformer, kdtree)

        X = df_split_feat[feature_cols]
        y_true = df_split_feat["pm25_monthly_mean"].to_numpy()
        y_pred = model.predict(X)

        report_metrics(split_name.upper(), y_true, y_pred)

    # Feature importance
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print("\nFeature importance (higher = used more):")
    for i, row in enumerate(imp_df.itertuples(index=False), start=1):
        print(f"{i:2d}. {row.feature:20s}  {row.importance:.4f}")

    # Hull from training sites
    train_site_points = [
        Point(float(lon), float(lat))
        for lat, lon in zip(train_sites["Latitude"], train_sites["Longitude"])
    ]
    hull = MultiPoint(train_site_points).convex_hull
    print("\nHull geom type:", hull.geom_type)

    # Output grid
    lats = np.arange(LAT_MIN, LAT_MAX + GRID_STEP_DEG, GRID_STEP_DEG, dtype=np.float32)
    lons = np.arange(LON_MIN, LON_MAX + GRID_STEP_DEG, GRID_STEP_DEG, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lons, lats)
    print("Grid shape (lat, lon):", lon2d.shape)

    mask = np.zeros(lon2d.shape, dtype=bool)
    for i in range(lon2d.shape[0]):
        for j in range(lon2d.shape[1]):
            mask[i, j] = hull.covers(Point(float(lon2d[i, j]), float(lat2d[i, j])))

    inside_cells = int(mask.sum())
    print("Inside-hull cells:", inside_cells)

    # Time axis
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

    # Prediction cube
    pred = np.full((len(times), len(lats), len(lons)), np.nan, dtype=np.float32)

    inside_lat = lat2d[mask].astype(np.float32)
    inside_lon = lon2d[mask].astype(np.float32)
    base_grid_df = pd.DataFrame({"Latitude": inside_lat, "Longitude": inside_lon})

    # Roads for grid points
    edges = load_or_download_roads_edges()

    motorway = edges[edges["is_motorway"]].copy()
    trunk = edges[edges["is_trunk"]].copy()
    primary = edges[edges["is_primary"]].copy()

    motorway_union = build_union_geometry(motorway)
    trunk_union = build_union_geometry(trunk)
    primary_union = build_union_geometry(primary)

    grid_pts_gdf = make_points_gdf(base_grid_df)

    print("\nComputing distance-to-road features for grid points...")
    base_grid_df["dist_motorway_km"] = dist_to_union_km(grid_pts_gdf, motorway_union)
    base_grid_df["dist_trunk_km"] = dist_to_union_km(grid_pts_gdf, trunk_union)
    base_grid_df["dist_primary_km"] = dist_to_union_km(grid_pts_gdf, primary_union)

    for c in ["dist_motorway_km", "dist_trunk_km", "dist_primary_km"]:
        base_grid_df[c] = base_grid_df[c].astype("float32").fillna(np.float32(999.0))

    base_grid_feat = add_spatial_features(base_grid_df, transformer, kdtree)

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
            "(projected x/y and distance to nearest training site), plus distance-to-road features from OSM "
            "(motorway/trunk/primary), and a spatial-blocked train/val/test split."
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
        "title": "Machine-learning-based gridded PM2.5 estimates (monthly) - spatially blocked model with roads",
        "summary": (
            "Monthly PM2.5 concentrations were estimated on a regular latitude-longitude grid using a "
            "Random Forest regression model trained on monitoring-site monthly averages. Spatial features "
            "include projected coordinates (EPSG:5070) and distance to nearest training monitoring site (km), "
            "plus distance-to-road features derived from OpenStreetMap (OSM) for motorway/trunk/primary roads, "
            "plus year and month. Train/validation/test splits were created using spatial blocking to better "
            "test generalization to new regions. Predictions are masked to the convex hull of training sites."
        ),
        "source": "Ground-based PM2.5 monitoring data (EPA) aggregated from daily to monthly means; roads from OpenStreetMap",
        "training_data": str(MONTHLY_PATH),
        "validation_strategy": f"Spatial blocking by {BLOCK_SIZE_KM} km blocks; blocks assigned to train/val/test.",
        "features": "x_m, y_m, dist_nn_km, dist_motorway_km, dist_trunk_km, dist_primary_km, year, month",
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
        "osm_bbox": f"{LAT_MIN},{LAT_MAX},{LON_MIN},{LON_MAX}",
        "osm_cache_edges": str(ROADS_EDGES_PARQUET),
    }

    encoding = {
        "pm25_pred": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan)
        }
    }

    ds.to_netcdf(out_nc, engine="netcdf4", encoding=encoding)

    print("\nSaved NetCDF:", out_nc)
    print("Done.")


if __name__ == "__main__":
    main()