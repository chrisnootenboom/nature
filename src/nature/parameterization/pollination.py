import logging
import typing

from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
from natcap.invest import utils
import pygeoprocessing

import nature


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
    """Creates a pandas df row for 'Solar Pollinator' that adapts the underlying land cover parameters to solar based on density

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


def prairie_restoration(prairie, lucode):
    """Creates a pandas df row for 'Prairie Restoration' that mimics the parameters in 'prairie' df

    Args:
        prairie (Pandas DataFrame): Single-row dataframe
        lucode (int): LULC code to assign Prairie Restoration

    Returns:
        Pandas DataFrame: Row with parameter values for the Prairie Restoration
    """
    # TODO Add exception test for single row: if not single row, tell user that the code is only selecting the first row
    result = prairie.head(1).copy()
    result["lucode"], result["lulc_name"] = lucode, "Prairie Restoration"
    return result


def buffer_strips(prairie, lucode):
    """Creates a pandas df row for 'Buffer Strips' that mimics the parameters in 'prairie' df

    Args:
        prairie (Pandas DataFrame): Single-row dataframe
        lucode (int): LULC code to assign Buffer Strips

    Returns:
        Pandas DataFrame: Row with parameter values for Buffer Strips
    """
    # TODO Add exception test for single row: if not single row, tell user that the code is only selecting the first row
    result = prairie.head(1).copy()
    result["lucode"], result["lulc_name"] = (
        lucode,
        "Buffer Strips / Vegetated Filter Strips",
    )
    return result


def flower_strips(parameter_table, lucode):
    """Creates a pandas df row with maximum values for nesting and floral parameters

    Args:
        parameter_table (Pandas DataFrame): DataFrame of all posible original parameters
        lucode (int): LULC code to assign Flower Strips

    Returns:
        Pandas DataFrame: Row with parameter values for Flower Strips
    """
    result = parameter_table.head(1).copy()
    for parameter in list(parameter_table):
        if parameter.startswith("nesting") or parameter.startswith("floral"):
            result[parameter] = parameter_table[parameter].max()
    result["lucode"], result["lulc_name"] = lucode, "Flower Strips"
    return result


def grassed_waterway(prairie, lucode):
    """Creates a pandas df row for 'Grassed Waterway' that mimics the parameters in 'prairie' df

    Args:
        prairie (Pandas DataFrame): Single-row dataframe
        lucode (int): LULC code to assign Grassed Waterway

    Returns:
        Pandas DataFrame: Row with parameter values for the Grassed Waterway
    """
    # TODO Add exception test for single row: if not single row, tell user that the code is only selecting the first row
    result = prairie.head(1).copy()
    result["lucode"], result["lulc_name"] = lucode, "Grassed Waterway"
    return result


def fertilizer_management(applicable_crops, lucode, m_fertilizer=1):
    # Creates pandas df rows that apply fertilization parameter downweights to all 'applicable crops' based on
    # literature review
    result = applicable_crops.copy()
    result["lucode"] += lucode
    result["lulc_name"] += f" - FERTILIZER MANAGEMENT"
    for parameter in list(result):
        if parameter.startswith("nesting") or parameter.startswith("floral"):
            result[parameter] = result[parameter] * m_fertilizer
    return result


def strips(applicable_crops, prairie, lucode, w_primary=0.9, w_strip=0.1):
    # Creates pandas df rows for STRIPS program that calculates weighted average of prairie restoration and crop
    # parameters based on STRIPS coverage (defaults to 10%)
    result = applicable_crops.copy()
    result["lucode"] += lucode
    result["lulc_name"] += f" - STRIPS"
    for parameter in list(result):
        if parameter.startswith("nesting") or parameter.startswith("floral"):
            result[parameter] = (
                result[parameter] * w_primary
                + prairie.reset_index().at[0, parameter] * w_strip
            )
    return result


def cover_crop(applicable_crops, cover_crop_df, lucode, w_primary=0.75, w_cover=0.25):
    # Creates pandas df rows for cover crop program that calculates weighted average of cover crop and primary crop
    # parameters based on cover crop coverage (defaults to 25%)
    result = applicable_crops.copy()
    result["lucode"] += lucode
    result[
        "lulc_name"
    ] += f" - COVER CROP {cover_crop_df.reset_index().at[0, 'lulc_name']}"
    for parameter in list(result):
        if parameter.startswith("nesting") or parameter.startswith("floral"):
            result[parameter] = (
                result[parameter] * w_primary
                + cover_crop_df.reset_index().at[0, parameter] * w_cover
            )
    return result


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
    borders_path = Path(work_dir) / "borders.shp"
    parcels_internal = parcels_gdf.copy()
    parcels_internal.geometry = parcels_gdf.geometry.buffer(-lulc_mean_pixel_size)
    borders = gpd.tools.overlay(parcels_gdf, parcels_internal, how="difference")
    borders.to_file(borders_path)

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
    nhd_project_path = Path(work_dir) / "nhd.shp"
    nhd_buffer_path = Path(work_dir) / "nhd_buffer.shp"

    nhd_gpd = pd.concat([gpd.read_file(file, mask=parcels_gdf) for file in nhd_list])
    nhd_gpd.to_file(nhd_project_path)

    print("project nhd")
    temp_dir = work_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    nhd_proj_path = nature.conditional_vector_project(
        nhd_project_path, lulc_info["projection_wkt"], work_dir
    )

    print("buffer nhd")
    nhd_proj_gpd = gpd.read_file(nhd_proj_path)
    nhd_buffer_gpd = nhd_proj_gpd.buffer(lulc_mean_pixel_size)
    nhd_buffer_gpd.to_file(nhd_buffer_path)

    # Burn buffers into lulc
    pygeoprocessing.rasterize(str(nhd_buffer_path), lulc, burn_values=[lucode])
