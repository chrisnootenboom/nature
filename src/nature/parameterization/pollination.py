import logging
import typing

from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
from natcap.invest import utils
import pygeoprocessing

from .. import nature
from . import param_utils


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

pathlike = str | Path


def solar_pollinators(
    land_cover_df,
    lucode,
    proportion_cavity_nesting=1 / 3,
    proportion_stem_nesting=1 / 3,
    proportion_ground_nesting=1 / 3,
    gcr=0.5,
):
    """Creates a pandas df row for 'Solar Pollinator' that adapts the underlying land cover parameters based on panel density (GCR)

    Args:
        land_cover_df (Pandas DataFrame): Single-row dataframe that represents the groundcover underneath the solar panels
        lucode (int): LULC code to assign Grassed Waterway

    Returns:
        Pandas DataFrame: Row with parameter values for the Solar Pollinator field
    """

    # TODO Calculate partial and full shade based on GCR, tilt angle, and distance between rows (from latitude??)

    # Creates a pandas df row for 'Solar Pollinator Plantings' that weights parameters from the 'prairie' df based on
    # literature review

    partial_shade = 1 - gcr
    full_shade = gcr

    result = land_cover_df.head(1).copy()
    result["lucode"], result["lulc_name"] = lucode, "Solar Pollinator Plantings"
    for parameter in list(result):
        if parameter.startswith("nesting"):
            result[parameter] = result[parameter] * (
                proportion_cavity_nesting * 1
                + proportion_stem_nesting * 1
                + proportion_ground_nesting * (partial_shade * 0.75 + full_shade * 0.05)
            )
        elif parameter.startswith("floral"):
            result[parameter] = result[parameter] * (
                partial_shade * 1.04 + full_shade * 1
            )
    return result


# TODO The following BMPs are not actually well-parameterized


# Replicating a single row BMP
def prairie_restoration(
    parameter_row_df: pd.DataFrame,
    lucode: int,
    label: str = param_utils.PRAIRIE_RESTORATION_LABEL,
):
    """Creates a pandas df row for 'Prairie Restoration' that mimics the input parameters

    Args:
        parameter_row_df (Pandas DataFrame): Single-row dataframe
        lucode (int): LULC code to assign Prairie Restoration
        label (str): (optional) Name of the BMP

    Returns:
        Pandas DataFrame: Row with parameter values for the Prairie Restoration
    """
    result = param_utils._replicate_parameter_row(parameter_row_df, lucode, label)
    return result


def buffer_strips(
    parameter_row_df: pd.DataFrame,
    lucode: int,
    label: str = param_utils.BUFFER_STRIP_LABEL,
):
    """Creates a pandas df row for 'Buffer Strips' that mimics the input parameters
    Args:
        prairie (Pandas DataFrame): Single-row dataframe
        lucode (int): LULC code to assign Buffer Strips
        label (str): (optional) Name of the BMP

    Returns:
        Pandas DataFrame: Row with parameter values for Buffer Strips
    """
    result = param_utils._replicate_parameter_row(parameter_row_df, lucode, label)

    return result


def hedgerows(
    parameter_row_df: pd.DataFrame,
    lucode: int,
    label: str = param_utils.HEDGEROW_LABEL,
):
    """Creates a pandas df row for 'Hedgerows' that mimics the input parameters
    Args:
        prairie (Pandas DataFrame): Single-row dataframe
        lucode (int): LULC code to assign Hedgerows
        label (str): (optional) Name of the BMP

    Returns:
        Pandas DataFrame: Row with parameter values for Buffer Strips
    """
    result = param_utils._replicate_parameter_row(parameter_row_df, lucode, label)

    return result


def grassed_waterway(
    parameter_row_df: pd.DataFrame,
    lucode: int,
    label: str = param_utils.GRASSED_WATERWAY_LABEL,
):
    """Creates a pandas df row for 'Grassed Waterway' that mimics the input parameters
    Args:
        prairie (Pandas DataFrame): Single-row dataframe
        lucode (int): LULC code to assign Buffer Strips
        label (str): (optional) Name of the BMP

    Returns:
        Pandas DataFrame: Row with parameter values for Grassed Waterway
    """
    result = param_utils._replicate_parameter_row(parameter_row_df, lucode, label)

    return result


# More complicated BMPs
def maximum_nesting_floral(
    parameter_table_df: pd.DataFrame,
    lucode: int,
    label: str = param_utils.MAX_PARAMETER_LABEL,
):
    """Creates a pandas df row with maximum values for nesting and floral parameters

    Args:
        parameter_table_df (Pandas DataFrame): DataFrame of all possible original parameters
        lucode (int): LULC code to assign Flower Strips
        label (str): (optional) Name of the BMP

    Returns:
        Pandas DataFrame: Row with parameter values for Flower Strips
    """
    result = parameter_table_df.head(1).copy()
    for parameter in list(parameter_table_df):
        if parameter.startswith("nesting") or parameter.startswith("floral"):
            result[parameter] = parameter_table_df[parameter].max()
    result["lucode"], result["lulc_name"] = lucode, label
    return result


def floral_strips(
    nesting_df: pd.DataFrame,
    floral_df: pd.DataFrame,
    lucode: int,
    label: str = param_utils.FLORAL_STRIP_LABEL,
):
    """Creates a pandas df row that takes nectar input from one parameter and floral input from another

    Args:
        nesting_df (Pandas DataFrame): DataFrame of nesting parameters
        floral_df (Pandas DataFrame): DataFrame of floral parameters
        lucode (int): LULC code to assign Flower Strips
        label (str): (optional) Name of the BMP

    Returns:
        Pandas DataFrame: Row with parameter values for Flower Strips
    """
    result = nesting_df.head(1).copy()
    for parameter in list(nesting_df):
        if parameter.startswith("nesting"):
            result[parameter] = nesting_df[parameter].max()
        elif parameter.startswith("floral"):
            result[parameter] = floral_df[parameter].max()
    result["lucode"], result["lulc_name"] = lucode, label
    return result


def fertilizer_management(applicable_crops, lucode, m_fertilizer=1):
    # Creates pandas df rows that apply fertilization parameter weights to all 'applicable crops' based on
    # literature review
    result = applicable_crops.copy()
    result["lucode"] += lucode
    result["lulc_name"] += f" - FERTILIZER MANAGEMENT"
    for parameter in list(result):
        if parameter.startswith("nesting") or parameter.startswith("floral"):
            result[parameter] = result[parameter] * m_fertilizer
    return result


def strips(
    applicable_crops_df: pd.DataFrame,
    parameter_row_df: pd.DataFrame,
    lucode_addition: int,
    label: str = param_utils.STRIPS_LABEL,
    w_strip: float = 0.1,
):
    """Creates a pandas df row for STRIPS programs that calculates weighted average of the restoration and crop
    parameters based on STRIPS coverage (defaults to 10% STRIPS coverage)

    Args:
        applicable_crops_df (Pandas DataFrame): DataFrame of crops that are applicable to the BMP
        parameter_table_df (Pandas DataFrame): DataFrame row of the cover crop used
        lucode_addition (int): LULC code to add to the base parameter (e.g. 1000 for cover crop)
        label (str): (optional) Name of the BMP
        w_cover (float): (optional) Weight of cover crop parameters (defaults to 25%)

    Returns:
        Pandas DataFrame: Row with parameter values for Cover Crop
    """

    w_primary = 1 - w_strip
    result = applicable_crops_df.copy()
    result["lucode"] += lucode_addition
    result["lulc_name"] += f" - {label}"
    for parameter in list(result):
        if parameter.startswith("nesting") or parameter.startswith("floral"):
            result[parameter] = (
                result[parameter] * w_primary
                + parameter_row_df.reset_index().at[0, parameter] * w_strip
            )
    return result


def cover_crop(
    applicable_crops_df: pd.DataFrame,
    parameter_row_df: pd.DataFrame,
    lucode_addition: int,
    label: str = param_utils.COVER_CROP_LABEL,
    w_cover: float = 0.25,
):
    """Creates a pandas df row that calculates weighted average of cover crop and primary crop
    parameters based on cover crop coverage (defaults to 25%)

    Args:
        applicable_crops_df (Pandas DataFrame): DataFrame of crops that are applicable to the BMP
        parameter_table_df (Pandas DataFrame): DataFrame row of the cover crop used
        lucode_addition (int): LULC code to add to the base parameter (e.g. 1000 for cover crop)
        label (str): (optional) Name of the BMP
        w_cover (float): (optional) Weight of cover crop parameters (defaults to 25%)

    Returns:
        Pandas DataFrame: Row with parameter values for Cover Crop
    """

    w_primary = 1 - w_cover
    result = applicable_crops_df.copy()
    result["lucode"] += lucode_addition
    result[
        "lulc_name"
    ] += f" - {label} {parameter_row_df.reset_index().at[0, 'lulc_name']}"
    for parameter in list(result):
        if parameter.startswith("nesting") or parameter.startswith("floral"):
            result[parameter] = (
                result[parameter] * w_primary
                + parameter_row_df.reset_index().at[0, parameter] * w_cover
            )
    return result


def seasonal_cover_crop(
    applicable_crops_df: pd.DataFrame,
    parameter_row_df: pd.DataFrame,
    lucode_addition: int,
    seasons: typing.List[str],
    label: str = param_utils.COVER_CROP_LABEL,
    w_cover: float = 0.25,
):
    """Creates a pandas df row that calculates weighted average of cover crop and primary crop
    parameters based on cover crop coverage (defaults to 25%) and the season of the cover crop
    (e.g. fall, winter, spring)

    Args:
        applicable_crops_df (Pandas DataFrame): DataFrame of crops that are applicable to the BMP
        parameter_table_df (Pandas DataFrame): DataFrame row of the cover crop used
        lucode_addition (int): LULC code to add to the base parameter (e.g. 1000 for cover crop)
        label (str): (optional) Name of the BMP
        w_cover (float): (optional) Weight of cover crop parameters (defaults to 25%)

    Returns:
        Pandas DataFrame: Row with parameter values for Cover Crop
    """

    w_primary = 1 - w_cover
    result = applicable_crops_df.copy()
    result["lucode"] += lucode_addition
    result[
        "lulc_name"
    ] += f" - {label} {parameter_row_df.reset_index().at[0, 'lulc_name']}"
    num_seasons = len(
        [parameter for parameter in list(result) if parameter.startswith("floral")]
    )
    w_season_cover = w_cover / num_seasons
    w_season_primary = 1 - w_season_cover
    for parameter in list(result):
        # Check if the parameter is a nesting parameter
        if parameter.startswith("nesting"):
            result[parameter] = (
                result[parameter] * w_season_primary
                + parameter_row_df.reset_index().at[0, parameter] * w_season_cover
            )
        # Check if the parameter is a floral parameter and if it is one of the seasons
        elif parameter.startswith("floral") and any(
            parameter.endswith(season) for season in seasons
        ):
            result[parameter] = (
                result[parameter] * w_primary
                + parameter_row_df.reset_index().at[0, parameter] * w_cover
            )
    return result


# Spatial stuff
def spatial_buffer_strips(lulc, parcels_gdf, lucode, work_dir):
    # Applies a spatial internal buffer to parcels to estimate field edge flower strips

    # Get cell size
    lulc_info = pygeoprocessing.get_raster_info(lulc)
    lulc_pixel_size_tuple = lulc_info["pixel_size"]
    try:
        lulc_mean_pixel_size = utils.mean_pixel_size_and_area(lulc_pixel_size_tuple)[0]
    except ValueError:
        lulc_mean_pixel_size = np.min(np.absolute(lulc_pixel_size_tuple))

    # Negative buffer at cell size then erase negative buffer from regular parcels
    borders_path = Path(work_dir) / "borders.gpkg"
    parcels_internal = parcels_gdf.copy()
    parcels_internal.geometry = parcels_gdf.geometry.buffer(-lulc_mean_pixel_size)
    borders = gpd.tools.overlay(parcels_gdf, parcels_internal, how="difference")
    borders.to_file(borders_path, driver="GPKG")

    # Burn borders into lulc
    pygeoprocessing.rasterize(str(borders_path), lulc, burn_values=[lucode])


def spatial_grassed_waterway(lulc, nhd_list, parcels_gdf, lucode, work_dir):
    # Applies a spatial buffer to NHD streams to estimate grassed waterway coverage

    # Get cell size
    lulc_info = pygeoprocessing.get_raster_info(lulc)
    lulc_pixel_size_tuple = lulc_info["pixel_size"]
    try:
        lulc_mean_pixel_size = utils.mean_pixel_size_and_area(lulc_pixel_size_tuple)[0]
    except ValueError:
        lulc_mean_pixel_size = np.min(np.absolute(lulc_pixel_size_tuple))

    # Subset NHD data
    print("read nhd")
    nhd_project_path = Path(work_dir) / "nhd.gpkg"
    nhd_buffer_path = Path(work_dir) / "nhd_buffer.gpkg"

    nhd_gpd = pd.concat([gpd.read_file(file, mask=parcels_gdf) for file in nhd_list])
    nhd_gpd.to_file(nhd_project_path, driver="GPKG")

    print("project nhd")
    temp_dir = work_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    nhd_proj_path = nature.conditional_vector_project(
        nhd_project_path, lulc_info["projection_wkt"], work_dir
    )

    print("buffer nhd")
    nhd_proj_gpd = gpd.read_file(nhd_proj_path)
    nhd_buffer_gpd = nhd_proj_gpd.buffer(lulc_mean_pixel_size)
    nhd_buffer_gpd.to_file(nhd_buffer_path, driver="GPKG")

    # Burn buffers into lulc
    pygeoprocessing.rasterize(str(nhd_buffer_path), lulc, burn_values=[lucode])
