"""Data loading and preprocessing module for TALEA satellite observations and indices.

Handles GeoJSON ingestion, tolerant JSON decoding, numerical parsing,
and multi-layer spatial alignment.
"""

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd


def load_geojson_lenient(source: Union[str, Path, bytes]) -> Dict[str, Any]:
    """Load a GeoJSON structure with tolerance for truncated or malformed trailing bytes.

    Parameters
    ----------
    source : Union[str, Path, bytes]
        File path, raw bytes, or string content of the GeoJSON.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON dictionary.
    """
    if isinstance(source, bytes):
        text = source.decode("utf-8", errors="replace")
    elif isinstance(source, Path) or (isinstance(source, str) and (Path(source).is_file() or "/" in source or "\\" in source)):
        path = Path(source)
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(source)
    else:
        text = str(source)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        last_brace = text.rfind("}")
        if last_brace != -1:
            sanitized = text[: last_brace + 1]
            return json.loads(sanitized)
        raise ValueError("Could not parse GeoJSON payload: no closing brace found.")


def extract_first_number(val: Any) -> Optional[float]:
    """Extract the first floating point or integer number from an arbitrary scalar.

    Parameters
    ----------
    val : Any
        Input string or numeric representation.

    Returns
    -------
    Optional[float]
        Extracted floating-point value, or None if no match is found.
    """
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    match = re.search(r"-?\d+(?:\.\d+)?", str(val))
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _read_json_from_source(data_source: Union[str, Path, zipfile.ZipFile], filename: str) -> Dict[str, Any]:
    """Read and decode a GeoJSON file from a directory or an open/closed ZipFile."""
    if isinstance(data_source, zipfile.ZipFile):
        names = data_source.namelist()
        target = filename
        if target not in names:
            candidates = [n for n in names if n.endswith(filename)]
            if not candidates:
                raise FileNotFoundError(f"File '{filename}' not found in archive.")
            target = candidates[0]
        raw_bytes = data_source.read(target)
        return load_geojson_lenient(raw_bytes)

    source_path = Path(data_source)
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path, "r") as zf:
            return _read_json_from_source(zf, filename)

    candidate_file = source_path / filename
    if not candidate_file.is_file():
        # Check subdirectories (e.g. data/)
        sub_candidate = source_path / "data" / filename
        if sub_candidate.is_file():
            candidate_file = sub_candidate
        else:
            raise FileNotFoundError(f"File '{filename}' not found in '{data_source}'.")

    return load_geojson_lenient(candidate_file)


def load_layer(
    data_source: Union[str, Path, zipfile.ZipFile],
    filename: str,
    colname: str,
) -> gpd.GeoDataFrame:
    """Load a single geospatial layer from GeoJSON, clean target attribute, and assign unique GID.

    Parameters
    ----------
    data_source : Union[str, Path, zipfile.ZipFile]
        Directory path or ZipFile archive containing the data files.
    filename : str
        Name of the GeoJSON file to load.
    colname : str
        Target attribute name for the layer's value column.

    Returns
    -------
    gpd.GeoDataFrame
        Geospatial dataframe containing geometry, cleaned value column, and 'gid' identifier.
    """
    geo_dict = _read_json_from_source(data_source, filename)
    features = geo_dict.get("features", [])
    gdf = gpd.GeoDataFrame.from_features(features)

    if "value" in gdf.columns:
        gdf = gdf.rename(columns={"value": colname})
    elif colname not in gdf.columns and len(gdf.columns) > 1:
        # Fallback to the first non-geometry column if neither 'value' nor colname exists
        attr_cols = [c for c in gdf.columns if c != "geometry"]
        if attr_cols:
            gdf = gdf.rename(columns={attr_cols[0]: colname})

    gdf[colname] = gdf[colname].apply(extract_first_number)
    gdf["gid"] = range(len(gdf))
    return gdf


def resolve_data_source(data_path: Union[str, Path] = "data.zip") -> Path:
    """Resolve the data source path between directory and zip archive fallback.

    Parameters
    ----------
    data_path : Union[str, Path]
        User-specified path to directory or zip file.

    Returns
    -------
    Path
        Resolved accessible path.
    """
    path = Path(data_path)
    if path.exists():
        return path

    # Fallback checks relative to current directory or parents
    fallbacks = [
        Path("data.zip"),
        Path("data"),
        Path(__file__).resolve().parent.parent.parent / "data.zip",
        Path(__file__).resolve().parent.parent.parent / "data",
    ]
    for fb in fallbacks:
        if fb.exists():
            return fb

    return path


def load_talea_dataset(
    data_path: Union[str, Path] = "data.zip",
    load_composite_indices: bool = True,
    ndvi_file: str = "ndvi_mean.geojson",
    albedo_file: str = "albedo_mean.geojson",
    zscore_file: str = "z_score.geojson",
    heat_veg_file: str = "heat_veg_index.geojson",
    heat_ret_file: str = "heat_retention_index.geojson",
) -> Tuple[gpd.GeoDataFrame, List[str], List[str]]:
    """Load, clean, and merge all satellite layers into a consolidated GeoDataFrame.

    Parameters
    ----------
    data_path : Union[str, Path], default="data.zip"
        Path to data directory or data.zip archive.
    load_composite_indices : bool, default=True
        Whether to attempt loading heat_veg_index and heat_retention_index.
    ndvi_file : str
        GeoJSON filename for mean NDVI layer.
    albedo_file : str
        GeoJSON filename for mean Albedo layer.
    zscore_file : str
        GeoJSON filename for normalized LST (z-score) target.
    heat_veg_file : str
        GeoJSON filename for vegetation heat index.
    heat_ret_file : str
        GeoJSON filename for heat retention index.

    Returns
    -------
    df_geo : gpd.GeoDataFrame
        Consolidated GeoDataFrame containing spatial polygons and all aligned indices.
    base_features : List[str]
        Names of baseline predictor columns (['ndvi_mean', 'albedo_mean']).
    extended_features : List[str]
        Names of all available predictor columns including composite indices.
    """
    resolved_source = resolve_data_source(data_path)

    ndvi = load_layer(resolved_source, ndvi_file, "ndvi_mean")
    albedo = load_layer(resolved_source, albedo_file, "albedo_mean")
    zscore = load_layer(resolved_source, zscore_file, "z_score")

    # Spatial-attribute merge on unique polygon index 'gid'
    df_geo = ndvi.merge(albedo[["gid", "albedo_mean"]], on="gid").merge(
        zscore[["gid", "z_score"]], on="gid"
    )

    base_features = ["ndvi_mean", "albedo_mean"]
    extended_features = list(base_features)

    if load_composite_indices:
        try:
            heat_veg = load_layer(resolved_source, heat_veg_file, "heat_veg_index")
            df_geo = df_geo.merge(heat_veg[["gid", "heat_veg_index"]], on="gid", how="left")
            extended_features.append("heat_veg_index")
        except FileNotFoundError:
            pass

        try:
            heat_ret = load_layer(resolved_source, heat_ret_file, "heat_retention_index")
            df_geo = df_geo.merge(heat_ret[["gid", "heat_retention_index"]], on="gid", how="left")
            extended_features.append("heat_retention_index")
        except FileNotFoundError:
            pass

    # Ensure strictly numeric casting and remove records missing essential targets or features
    for col in df_geo.columns:
        if col not in ("gid", "geometry"):
            df_geo[col] = pd.to_numeric(df_geo[col], errors="coerce")

    df_geo = df_geo.dropna(subset=["ndvi_mean", "albedo_mean", "z_score"]).reset_index(drop=True)

    return df_geo, base_features, extended_features
