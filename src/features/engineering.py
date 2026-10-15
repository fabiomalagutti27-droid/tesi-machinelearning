"""Feature engineering and spatial neighborhood calculations for urban surface analysis."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd


def compute_spatial_ndvi_buffers(
    gdf_ndvi: gpd.GeoDataFrame,
    distances: List[int] = [100, 200, 500],
    crs_metric: int = 32632,
    output_geojson: Optional[Union[str, Path]] = None,
    output_csv: Optional[Union[str, Path]] = None,
) -> gpd.GeoDataFrame:
    """Compute multi-scale spatial neighborhood NDVI averages across defined buffer radii.

    For each urban polygon, creates buffer geometries at specified metric distances,
    queries intersecting spatial neighbors via an R-tree spatial index, and computes
    the mean neighborhood NDVI.

    Parameters
    ----------
    gdf_ndvi : gpd.GeoDataFrame
        GeoDataFrame containing NDVI geometry and 'ndvi_mean' column.
    distances : List[int], default=[100, 200, 500]
        Radius distances in meters for spatial buffering.
    crs_metric : int, default=32632
        Target metric Projected Coordinate Reference System EPSG code (e.g. 32632 for UTM Zone 32N / Bologna).
    output_geojson : Optional[Union[str, Path]], default=None
        Optional path to persist resulting GeoJSON.
    output_csv : Optional[Union[str, Path]], default=None
        Optional path to persist resulting tabular CSV (excluding geometries).

    Returns
    -------
    gpd.GeoDataFrame
        Reprojected GeoDataFrame with added buffer feature columns ('ndvi_mean_{dist}m').
    """
    gdf = gdf_ndvi.copy()

    # Ensure a consistent geometric CRS in meters for accurate Euclidean buffering
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=crs_metric)

    # Spatial R-tree index for efficient intersection querying
    sindex = gdf.sindex

    for dist in distances:
        colname = f"ndvi_mean_{dist}m"
        buffer_means = []

        for geom in gdf.geometry:
            buffered_area = geom.buffer(dist)
            # Candidate filtering via bounding box intersection
            candidate_indices = list(sindex.intersection(buffered_area.bounds))
            candidates = gdf.iloc[candidate_indices]
            # Exact geometric intersection
            intersecting = candidates[candidates.intersects(buffered_area)]

            if not intersecting.empty and "ndvi_mean" in intersecting.columns:
                mean_val = intersecting["ndvi_mean"].mean()
            else:
                mean_val = None
            buffer_means.append(mean_val)

        gdf[colname] = buffer_means

    if output_geojson:
        gdf.to_file(output_geojson, driver="GeoJSON")
    if output_csv:
        gdf.drop(columns=["geometry"]).to_csv(output_csv, index=False)

    return gdf


def build_spatial_dataset(
    df_geo: gpd.GeoDataFrame,
    gdf_ndvi_buf: gpd.GeoDataFrame,
    distances: List[int] = [100, 200, 500],
) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """Align multi-scale spatial buffer features with base satellite predictors.

    Parameters
    ----------
    df_geo : gpd.GeoDataFrame
        Consolidated dataset containing 'gid', 'ndvi_mean', 'albedo_mean', 'z_score'.
    gdf_ndvi_buf : gpd.GeoDataFrame
        GeoDataFrame with computed 'ndvi_mean_{dist}m' columns.
    distances : List[int], default=[100, 200, 500]
        Buffer distances corresponding to computed columns.

    Returns
    -------
    df_dist : gpd.GeoDataFrame
        Merged dataset with spatial buffer predictors and target.
    spatial_features : List[str]
        List of predictor column names.
    """
    buffer_cols = [f"ndvi_mean_{d}m" for d in distances]
    cols_to_extract = ["gid"] + [c for c in buffer_cols if c in gdf_ndvi_buf.columns]

    if "gid" not in gdf_ndvi_buf.columns:
        gdf_ndvi_buf = gdf_ndvi_buf.copy()
        gdf_ndvi_buf["gid"] = range(len(gdf_ndvi_buf))

    df_dist = df_geo.merge(gdf_ndvi_buf[cols_to_extract], on="gid", how="inner")

    feature_cols = ["ndvi_mean", "albedo_mean"] + [c for c in buffer_cols if c in df_dist.columns]
    for c in feature_cols + ["z_score"]:
        df_dist[c] = pd.to_numeric(df_dist[c], errors="coerce")

    df_dist = df_dist.dropna(subset=feature_cols + ["z_score"]).reset_index(drop=True)
    return df_dist, feature_cols


def prepare_feature_matrices(
    df_geo: gpd.GeoDataFrame,
    feature_set: str = "base",
    spatial_buffer_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Extract predictor matrix X, response vector y, and feature column names.

    Parameters
    ----------
    df_geo : gpd.GeoDataFrame
        Dataset containing candidate features and target 'z_score'.
    feature_set : str, default="base"
        One of 'base' (NDVI + Albedo), 'extended' (base + composite indices),
        or 'spatial' (base + neighborhood buffer NDVI).
    spatial_buffer_cols : Optional[List[str]], default=None
        Custom spatial buffer column names if feature_set is 'spatial'.

    Returns
    -------
    X : pd.DataFrame
        Predictor matrix.
    y : pd.Series
        Target response vector (LST z-score).
    features : List[str]
        List of feature column names.
    """
    base_features = ["ndvi_mean", "albedo_mean"]

    if feature_set == "base":
        features = [c for c in base_features if c in df_geo.columns]
    elif feature_set == "extended":
        candidate_ext = base_features + ["heat_veg_index", "heat_retention_index"]
        features = [c for c in candidate_ext if c in df_geo.columns]
    elif feature_set == "spatial":
        buf_cols = spatial_buffer_cols or [
            c for c in df_geo.columns if c.startswith("ndvi_mean_") and c.endswith("m")
        ]
        features = [c for c in base_features if c in df_geo.columns] + buf_cols
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Must be 'base', 'extended', or 'spatial'.")

    clean_df = df_geo.dropna(subset=features + ["z_score"]).copy()
    X = clean_df[features].astype(float)
    y = clean_df["z_score"].astype(float)

    return X, y, features
