import logging
from pathlib import Path
import typing
from typing import List, Set, Dict, Tuple, Optional
import tempfile
import shutil
import pickle
import warnings

import requests
import xmltodict
from bs4 import BeautifulSoup

import math
import random
import pandas as pd
import numpy as np
import pint

import matplotlib.colors

import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
from osgeo import gdal, osr
import geopandas as gpd
from osgeo import ogr

import pygeoprocessing
from pygeoprocessing.geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
import pygeoprocessing.kernels
import natcap.invest.utils
import natcap.invest.spec_utils
import natcap.invest.validation

# from . import functions
# from . import rasterops

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

pathlike = str | Path

# TODO create singular logger file that applies via a logging python function (e.g. nature.logging.create_logger(level=logging.DEBUG))
# TODO this is so we can add collapsable messages for subsections of code (e.g. "message | submessage | subsubmessage")

# TODO Go through and clean up NoData entries, since I would like to internalize that parameter as much as possible.
# NOTE The main thing is to inherit NoData from the input raster. If changing raster dtype, then check if nodata fits in new dtype.


def message(*messages: str) -> str:
    # TODO LOGGER STUFF
    """Format a list of messages for logging.

    Args:
        *messages: Any number of messages to print.

    Returns:
        str: A formatted string of messages.
    """
    return " | ".join([str(message) for message in messages])


def geospatial_export(
    gdf: gpd.GeoDataFrame, file_name: str, workspace_path: Path
) -> Tuple[Path, Path]:
    """Export a GeoDataFrame to shapefile and geopackage.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to export.
        file_name (str): The name of the file to export.
        workspace_path (pathlib.Path): The directory to export to.

    Returns:
        tuple: The paths to the shapefile and geopackage.
    """
    output_shp = workspace_path / f"{file_name}.shp"
    output_gpkg = workspace_path / f"{file_name}.gpkg"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf.to_file(output_shp)
        gdf.to_file(output_gpkg, driver="GPKG")

    return output_shp, output_gpkg


def random_hex_color(num_colors: int = 1) -> list:
    """Generate a number of random hex colors.

    Args:
        num_colors (int): Number of colors to generate, defaults to 1.

    Returns:
        list: A list of random hex color strings.
    """

    return [
        "#" + "".join([random.choice("0123456789ABCDEF") for x in range(6)])
        for i in range(num_colors)
    ]


def strip_string(string: str) -> str:
    """Strip string of non-alphanumeric characters

    Args:
        string (str): String to strip

    Returns:
        str: Stripped string
    """
    return "".join([x if x.isalnum() else "_" if x == " " else "" for x in string])


def scenario_labeling(
    args: dict,
    baseline_args: list,
    baseline_scenario_label: str = "baseline",
) -> Tuple[list, dict]:
    """A function that takes all possible scenario-ized args, organizes them, duplicates baseline args if
    the scenario args do not exist, then produces a standard dict structure expected by all NCI Marginal Value
    models.

    Args:
        args (dict): The args dictionary used as input into the model.
            Must have an argument called 'scenario_{i}_list' for each baseline arg that is scenario-ized.
            Must have an argument called 'scenario_labels_list'.
        baseline_args (list): List of baseline args that become scenarios.
        scenario_labels_list (str): List of scenario suffixes.
        baseline_scenario_label (string, optional): String label for baseline results. Defaults to "baseline".

    Returns:
        tuple: scenario_name_list, scenario_args_dict
    """

    # First check if there are any scenario args to analyze, else return just the baseline args
    if any(
        [
            f"scenario_{x}_list" in args
            and args[f"scenario_{x}_list"] is not None
            and args[f"scenario_{x}_list"] != ""
            and len(args[f"scenario_{x}_list"]) > 0
            for x in baseline_args
        ]
    ):
        # Ensure all scenario args are in the args dict
        for baseline_arg in baseline_args:
            scenario_arg = f"scenario_{baseline_arg}_list"
            if (
                scenario_arg not in args
                or args[scenario_arg] is None
                or args[scenario_arg] == ""
            ):
                args[scenario_arg] = []

        # Used to check if any scenario args contain only one value. If so, we assume this value persists across all scenarios
        # and duplicate the single scenario arg across all scenarios
        n_scenarios = max(
            [
                len(args[f"scenario_{baseline_arg}_list"])
                for baseline_arg in baseline_args
            ]
        )

        # Identify scenario args and list them under their baseline arg name alongside the initial baseline arg
        scenario_items_dict = {
            baseline_arg: [args[baseline_arg]] + args[f"scenario_{baseline_arg}_list"]
            if len(args[f"scenario_{baseline_arg}_list"]) > 1
            else [args[baseline_arg]]
            + args[f"scenario_{baseline_arg}_list"] * n_scenarios
            for baseline_arg in baseline_args
        }

        # Duplicate any baseline arg if its corresponding scenario arg list does not exist
        n_scenarios = max([len(v) for v in scenario_items_dict.values()])
        scenario_items_dict = {
            baseline_arg: scenario_items
            if scenario_items is not None and len(scenario_items) > 1
            else [args[baseline_arg]] * n_scenarios
            for baseline_arg, scenario_items in scenario_items_dict.items()
        }

        # Ensure all scenario arg lists have the same length
        assert all(
            len(scenario_items)
            == len(scenario_items_dict[next(iter(scenario_items_dict))])
            for scenario_items in scenario_items_dict.values()
        ), f"All scenario args lists must have the same length. {', '.join([f'({baseline_arg}: {len(scenario_items)-1})' for baseline_arg, scenario_items in scenario_items_dict.items()])}"

        # Ensure baseline_scenario_label isn't in the scenario names
        assert (
            baseline_scenario_label not in args["scenario_labels_list"]
        ), f"Scenarios cannot be labeled '{baseline_scenario_label}'. Please rename the '{baseline_scenario_label}' scenario."

        # Ensure scenario labels exist for all scenario args and if not, relabel them all
        scenario_name_list = [baseline_scenario_label] + args["scenario_labels_list"]
        if len(scenario_name_list) != len(
            scenario_items_dict[next(iter(scenario_items_dict))]
        ):
            logger.error(
                f"Must provide the same number of scenario names ({len(scenario_name_list)-1}) as "
                f"scenario args ({len(scenario_items_dict[next(iter(scenario_items_dict))])-1}). "
                f"Relabeling as 'scenario1', 'scenario2', etc."
            )
            scenario_name_list = [baseline_scenario_label] + [
                f"scenario{i}"
                for i in range(
                    1,
                    len(scenario_items_dict[next(iter(scenario_items_dict))]),
                )
            ]

        # Create labeling schemata
        scenario_args_dict = {
            baseline_arg: dict(zip(scenario_name_list, scenario_items))
            for baseline_arg, scenario_items in scenario_items_dict.items()
        }
    else:
        # Return just the baseline args, but in the same format as the scenario args
        scenario_name_list = [baseline_scenario_label]
        scenario_args_dict = {
            baseline_arg: {baseline_scenario_label: args[baseline_arg]}
            for baseline_arg in baseline_args
        }

    return scenario_name_list, scenario_args_dict


def mosaic_rasters(
    raster_path_list: List[Path], output_path: Path, nodata_val: int | float
) -> None:
    """A function to mosaic a list of rasters

    Args:
        raster_list (list): List of pathlike rasters to mosaic
        output_path (pathlib.Path): Path to output mosaic raster

    Returns:
        None
    """

    raster_to_mosaic = []
    for p in raster_path_list:
        raster = rio.open(p)
        raster_to_mosaic.append(raster)
    mosaic, output_transform = merge(raster_to_mosaic)
    output_meta = raster.meta.copy()
    output_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output_transform,
            "nodata": nodata_val,
        }
    )
    with rio.open(output_path, "w", **output_meta) as m:
        m.write(mosaic)


def sjoin_largest_intersection(
    target_gdf: gpd.GeoDataFrame,
    join_gdf: gpd.GeoDataFrame,
    join_fields_list: list = None,
) -> gpd.GeoDataFrame:
    """Spatial join retaining information from the feature with the largest overlap

    Args:
        target_gdf (gpd.GeoDataFrame): GeoDataFrame to which to join
        join_gdf (gpd.GeoDataFrame): GeoDataFrame to join
        join_fields_list (optional, list): List of fields to join

    Returns:
        gpd.GeoDataFrame: Copy of target_gdf GeoDataFrame with joined fields
    """

    logger.debug(f"Spatial join (largest overlap)")

    # Create unique ID field for target layer to sort later
    target_fid_field = "_join_FID"
    try:
        target_fid_field not in target_gdf.columns
    except:
        logger.error(f"Field '{target_fid_field}' already exists in target layer")

    # Create unique area field for overlay layer to sort later
    area_field = "_area"
    try:
        area_field not in target_gdf.columns
    except:
        logger.error(f"Field '{area_field}' already exists in target layer")
    try:
        area_field not in join_gdf.columns
    except:
        logger.error(f"Field '{area_field}' already exists in join layer")

    # Identify join fields
    if join_fields_list is None:
        join_fields_list = join_gdf.columns.to_list()
        join_fields_list.remove("geometry")

    # Create copy of target gdf and assign unique index
    target_copy_gdf = target_gdf.copy()
    target_copy_gdf[target_fid_field] = target_copy_gdf.index

    # Intersect gdfs and calculate area
    overlay_gdf = gpd.overlay(target_copy_gdf, join_gdf, how="intersection")
    overlay_gdf[area_field] = overlay_gdf.geometry.area

    # Sort by area and drop duplicates
    overlay_gdf.sort_values(by=area_field, inplace=True)
    overlay_gdf.drop_duplicates(subset=target_fid_field, keep="last", inplace=True)
    overlay_gdf.drop(columns=[area_field], inplace=True)

    # Subset to join fields and set index
    overlay_gdf = overlay_gdf[join_fields_list + [target_fid_field]]
    overlay_gdf.set_index(target_fid_field, inplace=True)

    for join_field in join_fields_list:
        # Check if join field already exists in target layer
        target_fields = [field.lower() for field in target_gdf.columns]
        if join_field.lower() in target_fields:
            logger.warning(
                f"Field '{join_field}' may already exist in target layer. Renaming to '{join_field}_'"
            )
            # Rename join field to avoid duplicate names
            overlay_gdf.rename(columns={join_field: f"{join_field}_"}, inplace=True)

    # Join output to target gdf
    output_gdf = target_gdf.join(overlay_gdf)

    return output_gdf


def sjoin_representative_points(
    target_gdf: gpd.GeoDataFrame,
    join_gdf_list: List[gpd.GeoDataFrame],
    join_fields_list_list: List[List[str]],
) -> gpd.GeoDataFrame:
    """Spatial join using representative points

    Args:
        target_gdf (gpd.GeoDataFrame): GeoDataFrame to which to join
        join_gdf_list (list): List of GeoDataFrames to join
        join_fields_list_list (list): List of lists of fields to join

    Returns:
        gpd.GeoDataFrame: Copy of target_gdf GeoDataFrame with joined fields
    """

    logger.debug(f"Spatial join using representative points")
    # Create representative points
    target_points_gdf = target_gdf.copy()
    target_points_gdf.geometry = target_gdf.representative_point()

    # Iterate through join layers
    for join_gdf, join_fields_list in zip(join_gdf_list, join_fields_list_list):
        # Spatial join counties to representative points of parcels
        target_points_gdf = target_points_gdf.sjoin(
            join_gdf[join_fields_list + "geometry"], how="left"
        )
        target_points_gdf.drop("index_right", axis=1, inplace=True)

    output_gdf = gpd.GeoDataFrame(
        pd.merge(
            target_gdf,
            target_points_gdf[
                [
                    join_field
                    for join_field_list in join_fields_list_list
                    for join_field in join_field_list
                ]
            ],
            left_index=True,
            right_index=True,
        ),
        geometry="geometry",
    )

    return output_gdf


def nodata_validation(rasterio_dtype: str, nodata_val: int | float) -> int | float:
    """Validate nodata value for rasterio dtype, assigning new value if necessary.

    Args:
        rasterio_dtype (str): The rasterio dtype to validate against.
        nodata_val (int | float): The nodata value to validate.

    Returns:
        int | float: The validated nodata value.
    """
    # Automatically assign new nodata value if previous does not fit in new dtype
    output_nodata_val = (
        rio.dtypes.dtype_ranges[rasterio_dtype][
            1
        ]  # Selects largest possible value for nodata
        if not rio.dtypes.in_dtype_range(nodata_val, rasterio_dtype)
        else nodata_val
    )

    # if output_nodata_val in range(len(category_labels)):
    #     logger.warning("Nodata value does not fit within raster value map.")

    return output_nodata_val


def distance_to_pixels(
    landcover_raster_path: pathlike,
    units: pint.Unit = natcap.invest.spec_utils.u.meters,
) -> float:
    """Calculate the size of a raster pixel in terms of the specified units.

    Args:
        landcover_raster_path (pathlike): The path to the landcover raster.
        units (pint.Unit): The units to use for the pixel size. Defaults to meters.

    Returns:
        float: The size of a pixel in the specified units.
    """

    # Check that raster is in the specified unit
    projected_units_specs = {
        "projected": True,
        "projected_units": units,
    }

    try:
        # pass the entire arg spec into the validation function as kwargs
        # each type validation function allows extra kwargs with **kwargs
        warning_msg = natcap.invest.validation._VALIDATION_FUNCS["raster"](
            str(landcover_raster_path), **projected_units_specs
        )
        if warning_msg:
            logger.exception(warning_msg)
    except Exception:
        logger.exception(f"Error when validating raster for projected units ({units})")

    landcover_raster_info = pygeoprocessing.get_raster_info(str(landcover_raster_path))
    landcover_pixel_size_tuple = landcover_raster_info["pixel_size"]
    try:
        landcover_mean_pixel_size = natcap.invest.utils.mean_pixel_size_and_area(
            landcover_pixel_size_tuple
        )[0]
    except ValueError:
        landcover_mean_pixel_size = np.min(np.absolute(landcover_pixel_size_tuple))
        logger.debug(
            "Land Cover Raster has unequal x, y pixel sizes: "
            f"{landcover_pixel_size_tuple}. Using"
            f"{landcover_mean_pixel_size} as the mean pixel size."
        )
    return landcover_mean_pixel_size


def get_mean_pixel_size_and_area(raster_path: pathlike) -> Tuple[float, float]:
    """Get the mean pixel size and area of a raster.

    Args:
        raster_path (pathlike): The path to the raster.

    Returns:
        tuple: The mean pixel size and area of the raster.
    """

    raster_info = pygeoprocessing.get_raster_info(str(raster_path))
    pixel_size_tuple = raster_info["pixel_size"]
    try:
        mean_pixel_size, pixel_area = natcap.invest.utils.mean_pixel_size_and_area(
            pixel_size_tuple
        )
    except ValueError:
        mean_pixel_size = np.min(np.absolute(pixel_size_tuple))
        logger.debug(
            "Land Cover Raster has unequal x, y pixel sizes: "
            f"{pixel_size_tuple}. Using"
            f"{mean_pixel_size} as the mean pixel size."
        )
        pixel_area = mean_pixel_size**2
    return mean_pixel_size, pixel_area


def flat_disk_kernel_raster(kernel_filepath: pathlike, max_distance: int) -> None:
    """Create a flat disk kernel.

    The raster created will be a tiled GeoTiff, with 256x256 memory blocks.

    Args:
        kernel_filepath (pathlike): The path to the file on disk where this
            kernel should be stored.  If this file exists, it will be
            overwritten.
        max_distance (int): The distance (in pixels) of the
            kernel's radius.

    Returns:
        None

    """
    logger.info(
        f"Creating a disk kernel of distance {max_distance} at " f"{kernel_filepath}"
    )
    kernel_size = int(np.round(max_distance * 2 + 1))
    kernel_filepath = str(kernel_filepath)

    driver = gdal.GetDriverByName("GTiff")
    kernel_dataset = driver.Create(
        kernel_filepath.encode("utf-8"),
        kernel_size,
        kernel_size,
        1,
        gdal.GDT_Byte,
        options=["BIGTIFF=IF_SAFER", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"],
    )

    # Make some kind of geotransform and SRS. It doesn't matter what, but
    # having one will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(255)

    cols_per_block, rows_per_block = kernel_band.GetBlockSize()

    n_cols = kernel_dataset.RasterXSize
    n_rows = kernel_dataset.RasterYSize

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            # Numpy creates index rasters as ints by default, which sometimes
            # creates problems on 32-bit builds when we try to add Int32
            # matrices to float64 matrices.
            row_indices, col_indices = np.indices(
                (row_block_width, col_block_width), dtype=float
            )

            row_indices += float(row_offset - max_distance)
            col_indices += float(col_offset - max_distance)

            kernel_index_distances = np.hypot(row_indices, col_indices)
            kernel = kernel_index_distances < max_distance

            kernel_band.WriteArray(kernel, xoff=col_offset, yoff=row_offset)

    # Need to flush the kernel's cache to disk before opening up a new Dataset
    # object in interblocks()
    kernel_dataset.FlushCache()


def flat_square_kernel_raster(kernel_filepath: pathlike, max_distance: int) -> None:
    """Create a flat square kernel.

    The raster created will be a tiled GeoTiff, with 256x256 memory blocks.

    Args:
        kernel_filepath (pathlike): The path to the file on disk where this
            kernel should be stored.  If this file exists, it will be
            overwritten.
        max_distance (int): The distance (in pixels) of the kernel's radius.

    Returns:
        None

    """
    logger.info(
        f"Creating a disk kernel of distance {max_distance} at " f"{kernel_filepath}"
    )
    kernel_size = int(np.round(max_distance * 2 + 1))
    kernel_filepath = str(kernel_filepath)

    driver = gdal.GetDriverByName("GTiff")
    kernel_dataset = driver.Create(
        kernel_filepath.encode("utf-8"),
        kernel_size,
        kernel_size,
        1,
        gdal.GDT_Byte,
        options=["BIGTIFF=IF_SAFER", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"],
    )

    # Make some kind of geotransform and SRS. It doesn't matter what, but
    # having one will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(255)

    cols_per_block, rows_per_block = kernel_band.GetBlockSize()

    n_cols = kernel_dataset.RasterXSize
    n_rows = kernel_dataset.RasterYSize

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            # Numpy creates index rasters as ints by default, which sometimes
            # creates problems on 32-bit builds when we try to add Int32
            # matrices to float64 matrices.
            row_indices, col_indices = np.indices(
                (row_block_width, col_block_width), dtype=float
            )

            row_indices += float(row_offset - max_distance)
            col_indices += float(col_offset - max_distance)

            kernel_index_distances = np.hypot(row_indices, col_indices)
            kernel = kernel_index_distances > -1

            kernel_band.WriteArray(kernel, xoff=col_offset, yoff=row_offset)

    # Need to flush the kernel's cache to disk before opening up a new Dataset
    # object in interblocks()
    kernel_dataset.FlushCache()


def logistic_decay_kernel_raster(
    kernel_path: pathlike,
    terminal_distance: int | float,
    decay_start_distance: int | float = 0,
) -> None:
    """Create a raster-based exponential decay kernel.

    The raster created will be a tiled GeoTiff, with 256x256 memory blocks.

    Args:
        kernel_path (pathlike): The path to the file on disk where this
            kernel should be stored.  If this file exists, it will be
            overwritten.
        terminal_distance (int or float): The distance (in pixels) of the
            kernel's radius, the distance at which the value of the decay
            function starts to asymptote.
        decay_start_distance (int or float): The distance (in pixels) at which decay starts

    Returns:
        None
    """
    kernel_path = str(kernel_path)

    logger.debug("Creating logistic decay kernel")
    max_distance = terminal_distance + decay_start_distance
    kernel_size = int(np.round(max_distance * 2 + 1))

    driver = gdal.GetDriverByName("GTiff")
    kernel_dataset = driver.Create(
        kernel_path.encode("utf-8"),
        kernel_size,
        kernel_size,
        1,
        gdal.GDT_Float32,
        options=["BIGTIFF=IF_SAFER", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"],
    )

    # Make some kind of geotransform, it doesn't matter what but
    # will make GIS libraries behave better if it's all defined
    kernel_dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    kernel_dataset.SetProjection(srs.ExportToWkt())

    kernel_band = kernel_dataset.GetRasterBand(1)
    kernel_band.SetNoDataValue(-9999)

    cols_per_block, rows_per_block = kernel_band.GetBlockSize()

    n_cols = kernel_dataset.RasterXSize
    n_rows = kernel_dataset.RasterYSize

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    integration = 0.0
    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block

        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            # Numpy creates index rasters as ints by default, which sometimes
            # creates problems on 32-bit builds when we try to add Int32
            # matrices to float64 matrices.
            row_indices, col_indices = np.indices(
                (row_block_width, col_block_width), dtype=float
            )

            row_indices += float(row_offset - max_distance)
            col_indices += float(col_offset - max_distance)

            kernel_index_distances = np.hypot(row_indices, col_indices)

            # Use np vectorize to apply function
            kernel = np.where(
                kernel_index_distances > max_distance,
                0.0,
                np.vectorize(functions.parametric_generalized_logistic)(
                    kernel_index_distances,
                    a=1,
                    k=0,
                    b=terminal_distance,
                    m=decay_start_distance,
                ),
            )
            integration += np.sum(kernel)

            kernel_band.WriteArray(kernel, xoff=col_offset, yoff=row_offset)

    # Need to flush the kernel's cache to disk before opening up a new Dataset
    # object in interblocks()
    kernel_band.FlushCache()
    kernel_dataset.FlushCache()

    for block_data in pygeoprocessing.iterblocks(
        (str(kernel_path), 1), offset_only=True
    ):
        kernel_block = kernel_band.ReadAsArray(**block_data)
        kernel_block /= integration
        kernel_band.WriteArray(
            kernel_block, xoff=block_data["xoff"], yoff=block_data["yoff"]
        )

    kernel_band.FlushCache()
    kernel_dataset.FlushCache()
    kernel_band = None
    kernel_dataset = None


def convolve_2d_by_exponential(
    signal_raster_path: pathlike,
    target_convolve_raster_path: pathlike,
    decay_kernel_distance: int | float,
) -> None:
    """Convolve signal by an exponential decay of a given radius.

    Args:
        signal_rater_path (pathlike): path to single band signal raster.
        target_convolve_raster_path (pathlike): path to convolved raster.
        decay_kernel_distance (float): radius of 1/e cutoff of decay kernel
            raster in pixels.

    Returns:
        None.

    """

    # Ensure path arguments are Path objects
    signal_raster_path = Path(signal_raster_path)
    target_convolve_raster_path = Path(target_convolve_raster_path)

    logger.debug(
        f"Starting a convolution over {signal_raster_path} with a "
        f"decay distance of {decay_kernel_distance}"
    )
    temporary_working_dir = Path(
        tempfile.mkdtemp(dir=target_convolve_raster_path.parent)
    )
    exponential_kernel_path = temporary_working_dir / "exponential_decay_kernel.tif"
    pygeoprocessing.kernels.exponential_decay_kernel(
        str(exponential_kernel_path),
        decay_kernel_distance,
        decay_kernel_distance * 5,
    )
    pygeoprocessing.convolve_2d(
        (str(signal_raster_path), 1),
        (str(exponential_kernel_path), 1),
        str(target_convolve_raster_path),
        working_dir=str(temporary_working_dir),
        ignore_nodata_and_edges=True,
        mask_nodata=False,
    )
    shutil.rmtree(temporary_working_dir)


def convolve_2d_by_logistic(
    signal_raster_path: pathlike,
    target_convolve_raster_path: pathlike,
    terminal_kernel_distance: float,
    decay_start_distance: int | float = 0,
) -> None:
    """Convolve signal by an logistic decay of a given radius and start distance.

    Args:

        signal_rater_path (str): path to single band signal raster.
        target_convolve_raster_path (str): path to convolved raster.
        decay_kernel_distance (float): radius of 1/e cutoff of decay kernel
            raster in pixels.
        decay_start_distance (int or float): distance (in pixels) at which the logistic decay begins

    Returns:
        None.

    """

    # Ensure path arguments are Path objects
    signal_raster_path = Path(signal_raster_path)
    target_convolve_raster_path = Path(target_convolve_raster_path)

    logger.debug(
        f"Starting a logistic convolution over {signal_raster_path} with a "
        f"decay starting at {decay_start_distance} and terminating at {terminal_kernel_distance}"
    )
    temporary_working_dir = Path(
        tempfile.mkdtemp(dir=target_convolve_raster_path.parent)
    )
    logistic_kernel_path = temporary_working_dir / "logistic_decay_kernel.tif"
    logistic_decay_kernel_raster(
        logistic_kernel_path, terminal_kernel_distance, decay_start_distance
    )
    pygeoprocessing.convolve_2d(
        (str(signal_raster_path), 1),
        (str(logistic_kernel_path), 1),
        str(target_convolve_raster_path),
        working_dir=str(temporary_working_dir),
        ignore_nodata_and_edges=True,
        mask_nodata=False,
    )
    shutil.rmtree(temporary_working_dir)


def convolve_2d_by_flat(
    signal_raster_path: pathlike,
    target_convolve_raster_path: pathlike,
    terminal_distance: int | float,
    radial: bool = False,
    ignore_nodata_and_edges: bool = False,
    mask_nodata: bool = False,
    normalize: bool = True,
) -> None:
    # TODO check with the software team to see how to do a focal stats density function
    """Convolve signal by an flat kernel (either square or circular) with a
    defined radial distance.

    Args:

        signal_rater_path (str): path to single band signal raster.
        target_convolve_raster_path (str): path to convolved raster.
        terminal_distance (int or float): The distance (in pixels) of the
            kernel's radius, such that a 40-pixel distance creates an 80 by 80
            square.
        normalize (bool): Whether to normalize the kernel values to sum to 1.
        radial (bool): Whether to create a radial kernel (True) or a square
            (False, default).

    Returns:
        None.

    """

    # Ensure path arguments are Path objects
    signal_raster_path = Path(signal_raster_path)
    target_convolve_raster_path = Path(target_convolve_raster_path)

    logger_message = (
        f"Starting a {int(terminal_distance)}-pixel circular flat convolution over {signal_raster_path}"
        if radial
        else f"Starting a {int(terminal_distance)}-pixel flat square convolution over {signal_raster_path}"
    )
    logger.debug(logger_message)
    with tempfile.NamedTemporaryFile(
        prefix="flat_kernel",
        delete=False,
        suffix=".tif",
        dir=target_convolve_raster_path.parent,
    ) as flat_kernel_file:
        flat_kernel_path = Path(flat_kernel_file.name)

    if radial:
        flat_disk_kernel_raster(flat_kernel_path, terminal_distance)
    else:
        flat_square_kernel_raster(flat_kernel_path, terminal_distance)
    with tempfile.TemporaryDirectory(
        prefix="flat_kernel", dir=target_convolve_raster_path.parent
    ) as temporary_working_dir:
        pygeoprocessing.convolve_2d(
            (str(signal_raster_path), 1),
            (str(flat_kernel_path), 1),
            str(target_convolve_raster_path),
            working_dir=str(temporary_working_dir),
            ignore_nodata_and_edges=ignore_nodata_and_edges,
            mask_nodata=mask_nodata,
            normalize_kernel=normalize,
        )
    shutil.rmtree(flat_kernel_path)


def booleanize_raster(
    input_raster: pathlike, output_raster: Path, raster_value_list: list
) -> None:
    """A function to reclassify a raster to boolean based on a list of raster values
    to be considered True.

    Args:
        input_raster (pathlike): Path to the input raster
        output_raster (pathlike): Path to the output raster
        raster_value_list (list): List of raster values to reclassify to 1

    Returns:
        None
    """
    # Ensure path arguments are Path objects
    input_raster = Path(input_raster)
    output_raster = Path(output_raster)

    logger.info(f"Reclassifying {input_raster.stem} to boolean")
    raster_gdal = gdal.Open(str(input_raster))
    raster_values = np.unique(np.array(raster_gdal.GetRasterBand(1).ReadAsArray()))

    pygeoprocessing.reclassify_raster(
        (str(input_raster), 1),
        {code: (1 if code in raster_value_list else 0) for code in raster_values},
        str(output_raster),
        gdal.GDT_Byte,
        0,
    )


def reclassify_raster_from_dataframe(
    input_raster: pathlike,
    output_raster: pathlike,
    reclass_df: pd.DataFrame,
    input_column: str,
    output_column: str,
) -> None:
    """A function to reclassify a raster based on a dataframe

    Args:
        input_raster (pathlike): Path to the input raster
        output_raster (pathlike): Path to the output raster
        reclass_df (pd.DataFrame): DataFrame with input and output columns
        input_column (str): Name of input column in reclass_df
        output_column (str): Name of output column in reclass_df

    Returns:
        None
    """
    # Ensure path arguments are Path objects
    input_raster = Path(input_raster)
    output_raster = Path(output_raster)

    logger.info(f"Reclassifying {input_raster.stem}")
    raster_gdal = gdal.Open(str(input_raster))
    raster_values = np.unique(np.array(raster_gdal.GetRasterBand(1).ReadAsArray()))

    reclass_df.set_index(input_column, inplace=True)

    pygeoprocessing.reclassify_raster(
        (str(input_raster), 1),
        {
            code: (
                reclass_df[output_column][code] if code in reclass_df.index else code
            )
            for code in raster_values
        },
        str(output_raster),
        pygeoprocessing.get_raster_info(str(input_raster))["datatype"],
        pygeoprocessing.get_raster_info(str(input_raster))["nodata"][0],
    )


def extract_cdl(
    year: int, fips: int, temp_dir: pathlike, nodata_val: int | float = 0
) -> Path:
    """Extract CDL data for a given year and FIPS code

    Args:
        year (int): year to extract
        fips (int): FIPS code to extract
        temp_dir (str or pathlib.Path): temporary directory to store data

    Returns:
        pathlib.Path: path to extracted CDL data
    """

    logger.info(f"Extracting {year} CDL data for FIPS {fips}")

    # Ensure path arguments are Path objects
    temp_dir = Path(temp_dir)

    # Get CDL data from NASS as XML link
    nass_xml_url = rf"https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year={year}&fips={fips}"
    r = requests.get(nass_xml_url)

    # Extract XML link from XML
    soup = BeautifulSoup(r.content, "xml")
    nass_xml_return_url = soup.find("returnURL").text
    # nass_xml_return_url = xmltodict.parse(r.content)["ns1:GetCDLFileResponse"]["returnURL"]  # Old method

    # Download CDL data to local file
    r = requests.get(nass_xml_return_url, allow_redirects=True)
    outfile = temp_dir / f"CDL_{year}_{fips}.tif"
    with open(outfile, "wb") as cdl_out:
        cdl_out.write(r.content)

    # Update nodata value to 0
    with rio.open(outfile, "r+") as dataset:
        dataset.nodata = nodata_val

    return outfile


def copy_raster_to_new_datatype(
    raster_path: pathlike,
    output_raster_path: pathlike,
    rasterio_dtype: str = rio.int16,
    nodata_val: int | float = None,
) -> None:
    """A function to copy a raster to a new datatype

    Args:
        raster_path (str or pathlib.Path): Path to the input raster
        output_raster_path (pathlib.Path): Path to the output raster

    Returns:
        None
    """
    # Force pathlib compliance
    raster_path = Path(raster_path)
    output_raster_path = Path(output_raster_path)

    src = rio.open(raster_path)
    src_array = src.read(1)
    profile = src.profile

    # Update profile. If nodata_val is not None, set nodata value to nodata_val
    if nodata_val is not None:
        profile.update(dtype=rasterio_dtype, count=1, compress="lzw", nodata=nodata_val)
    else:
        profile.update(dtype=rasterio_dtype, count=1, compress="lzw")

    with rio.open(output_raster_path, "w", **profile) as dst:
        dst.write(src_array.astype(rasterio_dtype), 1)


def rename_invest_results(invest_model: str, invest_args: dict, suffix: str) -> None:
    """Rename InVEST results to include the suffix.

    Args:
        invest_model (str): InVEST model name.
        invest_args (dict): InVEST model arguments.
        suffix (str): String to append to the end of the file name.

    Returns:
        None
    """
    results_suffix = natcap.invest.utils.make_suffix_string(
        invest_args, "results_suffix"
    )

    # Modify relevant outputs
    # TODO Change pollination results to be dependent on pollinator species
    if invest_model == "pollination":
        result_file_list = [Path(invest_args["workspace_dir"]) / f"avg_pol_abd.tif"]
    elif invest_model == "pollination_mv":
        result_file_list = [
            Path(invest_args["workspace_dir"])
            / f"marginal_value_general{results_suffix}.tif"
        ]
    elif invest_model == "sdr":
        result_file_list = [
            Path(invest_args["workspace_dir"]) / f"sed_export{results_suffix}.tif",
            Path(invest_args["workspace_dir"])
            / f"watershed_results_sdr{results_suffix}.shp",
        ]
    elif invest_model == "ndr":
        result_file_list = [
            Path(invest_args["workspace_dir"]) / f"n_total_export{results_suffix}.tif",
            Path(invest_args["workspace_dir"])
            / f"p_surface_export{results_suffix}.tif",
            Path(invest_args["workspace_dir"])
            / f"watershed_results_ndr{results_suffix}.gpkg",
        ]
    elif invest_model == "carbon":
        result_file_list = [
            Path(invest_args["workspace_dir"]) / f"tot_c_cur{results_suffix}.tif",
            Path(invest_args["workspace_dir"]) / f"tot_c_fut{results_suffix}.tif",
            Path(invest_args["workspace_dir"]) / f"delta_cur_fut{results_suffix}.tif",
        ]

    for result_file in result_file_list:
        if result_file.exists():
            result_file.rename(
                Path(invest_args["workspace_dir"])
                / f"{result_file.stem}_{suffix}{result_file.suffix}"
            )


def clip_raster_by_vector(
    mask_path: pathlike,
    raster_path: pathlike,
    output_raster_path: pathlike,
    field_filter: tuple = None,
) -> None:
    """Clip a raster by a vector mask.

    Args:
        mask_path (str): Path to the mask vector.
        raster_path (str): Path to the raster to be clipped.
        output_raster_path (str): Path to the output raster.
        field_filter (tuple(str, list)): tuple of field string and value list
            for filtering the vector

    Returns:
        None
    """
    mask_gdf = gpd.read_file(mask_path)

    clip_raster_by_gdf(mask_gdf, raster_path, output_raster_path, field_filter)


def clip_raster_by_gdf(
    mask_gdf: gpd.GeoDataFrame,
    raster_path: pathlike,
    output_raster_path: pathlike,
    field_filter: tuple = None,
) -> None:
    """Clip a raster by a vector gdf mask.

    Args:
        mask_gdf (GeoDataFrame): Geopandas GeoDataFrame.
        raster_path (pathlike): Path to the raster to be clipped.
        output_raster_path (pathlike): Path to the output raster.
        field_filter (tuple(str, list)): tuple of field string and value list
            for filtering the vector

    Returns:
        None
    """

    if field_filter is not None:
        mask_gdf = mask_gdf[mask_gdf[field_filter[0]].isin(field_filter[1])]

    with rio.open(raster_path) as src:
        mask_gdf = mask_gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, mask_gdf.geometry, crop=True)
        out_meta = src.meta.copy()  # copy the metadata of the source raster

    out_meta.update(
        {
            "driver": "Gtiff",
            "height": out_image.shape[1],  # height starts with shape[1]
            "width": out_image.shape[2],  # width starts with shape[2]
            "transform": out_transform,
        }
    )

    with rio.open(output_raster_path, "w", **out_meta) as dst:
        dst.write(out_image)


def burn_polygon_add(
    vector_path: pathlike,
    base_raster_path: pathlike,
    intermediate_workspace_path: pathlike,
    target_raster_path: pathlike,
    burn_value: int | float,
    rasterio_dtype=rio.int8,
    nodata_val: int | float = None,
    burn_add: bool = True,
    burn_attribute: str = None,
    constraining_raster_value_tuple: typing.Tuple[
        pathlike, typing.List[int | float]
    ] = None,
) -> None:
    """Burns a polygon into a raster by adding a specified value.

    Args:
        vector_path (pathlike): Path to the vector to be burned.
        base_raster_path (pathlike): Path to the raster to be burned.
        intermediate_workspace_path (pathlike): Path to the intermediate workspace.
        target_raster_path (pathlike): Path to the output raster.
        burn_value (int): Value to be burned into the raster.
        rasterio_dtype (rasterio.dtype, optional): Rasterio datatype to use for the output raster.
        nodata_val (int/float, optional): Value to use for nodata. If None, will use the nodata value of the base raster.
        burn_add (bool, optional): Whether to add the burn value to the existing raster values. If False, will overwrite.
        burn_attribute (str, optional): Attribute to use for burning. If None, will burn all polygons with the same value.
        constraining_raster_value_tuple (tuple, optional): Tuple of (pathlike, List[int/float])
            to use for constraining the output raster. If None, will not constrain.

    Returns:
        None
    """
    # TODO make a temp dir for the _burned_ raster

    # Make all path variables Path objects
    vector_path = Path(vector_path)
    base_raster_path = Path(base_raster_path)
    intermediate_workspace_path = Path(intermediate_workspace_path)

    # Get raster info
    base_raster_info = pygeoprocessing.get_raster_info(str(base_raster_path))

    assert (
        base_raster_info["nodata"][0] is not None
    ), f"Warning: base_raster_path ({base_raster_path}) does not have a defined NoData value."

    # Dissolve vector to single feature to avoid overlapping polygons
    logger.debug(f"Dissolving burn vector")
    vector_gdf = gpd.read_file(vector_path)
    if burn_attribute is not None:
        vector_gdf = vector_gdf.dissolve(by=burn_attribute)
    else:
        vector_gdf = vector_gdf.dissolve()
    dissolved_vector_path = (
        intermediate_workspace_path / f"_dissolved_{vector_path.stem}.gpkg"
    )
    vector_gdf.to_file(dissolved_vector_path)

    # Copy raster to different datatype to accommodate new integer values
    intermediate_raster_path = (
        intermediate_workspace_path / f"_burned_{Path(base_raster_path).stem}.tif"
    )
    if nodata_val is not None:
        copy_raster_to_new_datatype(
            base_raster_path,
            intermediate_raster_path,
            rasterio_dtype=rasterio_dtype,
            nodata_val=nodata_val,
        )
    else:
        copy_raster_to_new_datatype(
            base_raster_path,
            intermediate_raster_path,
            rasterio_dtype=rasterio_dtype,
        )

    # Get raster info for intermediate raster
    target_raster_info = pygeoprocessing.get_raster_info(str(intermediate_raster_path))

    # Set burn options
    option_list = ["MERGE_ALG=ADD"] if burn_add else []

    # Burn and add vector to raster values
    logger.debug(f"Burning vector into raster")
    if burn_attribute is not None:
        pygeoprocessing.rasterize(
            str(dissolved_vector_path),
            str(intermediate_raster_path),
            burn_values=[burn_value],
            option_list=option_list + [f"ATTRIBUTE={burn_attribute}"],
        )
    else:
        pygeoprocessing.rasterize(
            str(dissolved_vector_path),
            str(intermediate_raster_path),
            burn_values=[burn_value],
            option_list=option_list,
        )

    # Constraining output by constraint raster, if called for
    if constraining_raster_value_tuple is not None:
        # Create constraining raster calculator function
        def _constrain_op(
            base_array, change_array, constraining_array, *constraining_values
        ):
            out_array = np.where(
                np.isin(constraining_array, constraining_values),
                change_array,
                base_array,
            )

            return out_array

        # Replace values in intermediate raster with original raster values where constrained
        logger.debug(f"Constraining burn by raster")
        constrained_raster_path = (
            intermediate_workspace_path
            / f"_constrained_burned_{Path(base_raster_path).stem}.tif"
        )
        constraint_value_list = [
            (constraint, "raw") for constraint in constraining_raster_value_tuple[1]
        ]
        pygeoprocessing.raster_calculator(
            [
                (str(base_raster_path), 1),
                (str(intermediate_raster_path), 1),
                (str(constraining_raster_value_tuple[0]), 1),
            ]
            + constraint_value_list,
            _constrain_op,
            str(constrained_raster_path),
            target_raster_info["datatype"],
            target_raster_info["nodata"][0],
        )

        # Update intermediate raster path for final processing step
        intermediate_raster_path = constrained_raster_path

    # Remove areas where vector burned over nodata
    logger.debug(f"Removing nodata areas")

    def mask_op(base_array, mask_array, nodata):
        result = np.copy(base_array)
        result[mask_array == nodata] = nodata
        return result

    pygeoprocessing.raster_calculator(
        [
            (str(intermediate_raster_path), 1),
            (str(base_raster_path), 1),
            (base_raster_info["nodata"][0], "raw"),
        ],
        mask_op,
        str(target_raster_path),
        target_raster_info["datatype"],
        target_raster_info["nodata"][0],
    )


def overlay_on_top_of_lulc(
    vector_path: pathlike,
    base_lulc_path: pathlike,
    intermediate_workspace_path: pathlike,
    target_raster_path: pathlike,
    burn_value: int | float = 1,
    nodata_val: int | float = None,
    burn_add: bool = True,
    burn_attribute: str = None,
    constraining_raster_value_tuple: Tuple[pathlike, List[int | float]] = None,
) -> None:
    """Burns a polygon into a raster by adding a specified value, ensuring that the burn values are using digits unused in the base raster.

    Args:
        vector_path (pathlike): Path to the vector to be burned.
        base_raster_path (pathlike): Path to the raster to be burned.
        intermediate_workspace_path (pathlike): Path to the intermediate workspace.
        target_raster_path (pathlike): Path to the output raster.
        burn_value (int): Value to be burned into the raster.
        nodata_val (int/float): Value to use for nodata. If None, will use the nodata value of the base raster.
        burn_add (bool): Whether to add the burn value to the existing raster values. If False, will overwrite.
        burn_attribute (str): Attribute to use for burning. If None, will burn all polygons with the same value.
        constraining_raster_value_tuple (tuple, optional): Tuple of (pathlike, List[int/float])
            to use for constraining the output raster. If None, will not constrain.

    Returns:
        None
    """

    # Extract maximum value of the raster to determine how many digits are needed
    with rio.open(base_lulc_path) as dataset:
        lulc_digits = len(str(int(dataset.statistics(1).max)))

    base_nodata_val = (
        pygeoprocessing.get_raster_info(str(base_lulc_path))["nodata"][0]
        if nodata_val is None
        else nodata_val
    )

    # If an attribute is specified, dissolve the vector by that attribute and multiply the attribute values by 10**lulc_digits
    if burn_attribute is not None:
        vector_gdf = gpd.read_file(vector_path)
        # TODO check if burn_attribute is in vector_gdf.columns
        vector_gdf[burn_attribute] = (
            vector_gdf[burn_attribute].astype(int) * 10**lulc_digits
        )
        intermediate_vector_path = (
            intermediate_workspace_path / f"_dissolved_{vector_path.stem}.gpkg"
        )
        vector_gdf.to_file(intermediate_vector_path)
        # Get the minimum raster datatype possible to house the digits required for this overlay
        rasterio_dtype = rio.dtypes.get_minimum_dtype(
            [vector_gdf[burn_attribute].max(), base_nodata_val]
        )
    else:
        intermediate_vector_path = vector_path
        rasterio_dtype = rio.dtypes.get_minimum_dtype(
            [burn_value * 10**lulc_digits, base_nodata_val]
        )

    burn_polygon_add(
        intermediate_vector_path,
        base_lulc_path,
        intermediate_workspace_path,
        target_raster_path,
        burn_value=burn_value,
        rasterio_dtype=rasterio_dtype,
        nodata_val=nodata_val,
        burn_add=burn_add,
        burn_attribute=burn_attribute,
        constraining_raster_value_tuple=constraining_raster_value_tuple,
    )


def calculate_load_per_pixel(
    lulc_raster_path: pathlike,
    lucode_to_parameters: dict,
    target_load_raster: pathlike,
    nodata_val: float,
) -> None:
    """Calculate load raster by mapping landcover and multiplying by area.

    Args:
        lulc_raster_path (pathlike): path to integer landcover raster. Must be projected in meters!
        lucode_to_parameters (dict): a dictionary mapping landcover IDs to a per-hectare chemical load.
        target_load_raster (string): path to target raster that will have
            total load per pixel.
        nodata_val (float): nodata value to use in target raster.

    Returns:
        None.

    """
    # TODO ensure that lulc_raster_path is projected in meters
    lulc_raster_info = pygeoprocessing.get_raster_info(str(lulc_raster_path))
    nodata_landuse = lulc_raster_info["nodata"][0]
    cell_area_ha = abs(np.prod(lulc_raster_info["pixel_size"])) * 0.0001

    def _map_load_op(lucode_array):
        """Convert unit load to total load & handle nodata."""
        result = np.empty(lucode_array.shape)
        result[:] = nodata_val
        for lucode in np.unique(lucode_array):
            if lucode != nodata_landuse:
                try:
                    result[lucode_array == lucode] = (
                        lucode_to_parameters[lucode] * cell_area_ha
                    )
                except KeyError:
                    raise KeyError(
                        "lucode: %d is present in the landuse raster but "
                        "missing from the biophysical table" % lucode
                    )
        return result

    pygeoprocessing.raster_calculator(
        [(str(lulc_raster_path), 1)],
        _map_load_op,
        str(target_load_raster),
        gdal.GDT_Float32,
        nodata_val,
    )


def categorize_raster(
    raster_path: pathlike,
    output_raster_path: pathlike,
    category_bins: List[float],
    category_string_labels: List[str],
    category_hex_colors: List[str],
    raster_band_label: str,
    category_value_labels: List[int] = None,
) -> None:
    """Bin raster into a smaller number of integer classes and assign labels and colors to each class.


    Args:
        raster_path (str): path to input raster
        output_raster_path (str): path to output raster
        raster_band_label (int): band number of input raster to use
        category_bins (list): list of bin edges
        category_string_labels (list): list of labels for each bin
        category_colors (list): list of colors for each bin
        category_value_labels (list): (optional) list of values for each bin

    Returns:
        None
    """

    if category_value_labels is None:
        # Default to categories starting at 1
        category_value_labels = [
            i + 1 for i in list(range(len(category_string_labels)))
        ]
        raster_type = rio.dtypes.get_minimum_dtype(category_value_labels)
        output_nodata_val = nodata_validation(raster_type, 0)
    else:
        input_nodata_val = pygeoprocessing.get_raster_info(str(raster_path))["nodata"][
            0
        ]
        raster_type = rio.dtypes.get_minimum_dtype(category_value_labels)
        output_nodata_val = nodata_validation(raster_type, input_nodata_val)
        if output_nodata_val in category_value_labels:
            logger.warning("Nodata value does not fit within raster value map.")

    # Check if input raster nodata value is within the range of the output raster type and adapt if necessary

    # Create constraining raster calculator function
    def _bin_op(array, bins):
        """Bin and label rasters."""
        valid_mask = ~natcap.invest.utils.array_equals_nodata(array, input_nodata_val)
        result = np.empty_like(array)
        result[:] = output_nodata_val
        result[valid_mask] = np.array(
            pd.cut(array[valid_mask], bins, labels=category_value_labels)
        )
        return result

    # Replace values in intermediate raster with original raster values where constrained
    logger.debug(f"Binning raster {Path(raster_path).stem}")
    pygeoprocessing.raster_calculator(
        [
            (str(raster_path), 1),
            (category_bins, "raw"),
        ],
        _bin_op,
        str(output_raster_path),
        rio.dtypes._get_gdal_dtype(raster_type),
        output_nodata_val,
    )

    assign_raster_labels_and_colors(
        output_raster_path,
        category_value_labels,
        category_string_labels,
        category_hex_colors,
        raster_band_label,
    )


def assign_raster_labels_and_colors(
    raster_path: pathlike,
    category_values: List[int],
    category_labels: List[str],
    category_hex_colors: List[str],
    raster_band_label: str,
) -> None:
    """Assign labels and colors to raster categories.

    Args:
        raster_path (str): path to input raster
        category_values (list): list of category values (must be in ascending order!)
        category_labels (list): list of labels for each category value
        category_colors (list): list of colors for each category value
        raster_band_label (int): Band label

    Returns:
        None
    """
    # TODO add validation error if category_values, category_labels, and category_hex_colors are not the same length
    # Read raster band
    raster = gdal.OpenEx(str(raster_path), gdal.GA_Update | gdal.OF_RASTER)
    band = raster.GetRasterBand(1)

    # # Raster Attribute Table
    rat = gdal.RasterAttributeTable()

    rat.CreateColumn("Value", gdal.GFT_Integer, gdal.GFU_MinMax)
    rat.CreateColumn("Label", gdal.GFT_String, gdal.GFU_Name)

    for i, (value, label) in enumerate(zip(category_values, category_labels)):
        rat.SetValueAsInt(i, 0, int(value))
        rat.SetValueAsString(i, 1, str(label))

    band.SetDefaultRAT(rat)

    # GDAL Color Table
    color_table = gdal.ColorTable()

    for value, color in zip(category_values, category_hex_colors):
        color_table.SetColorEntry(
            int(value), tuple(int(c * 255) for c in matplotlib.colors.to_rgba(color))
        )

    # set color table and color interpretation
    band.SetRasterColorTable(color_table)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    # # set category names
    dtype = rio.dtypes.dtype_fwd[band.DataType]
    if dtype != "uint8":
        logger.warning(
            "Raster data type is not uint8. Assigning category names may increase raster size."
        )
    raster_values = rio.dtypes.dtype_ranges[rio.dtypes.dtype_fwd[band.DataType]]
    value_dict = {
        value: label for value, label in zip(category_values, category_labels)
    }
    full_labels = [
        value_dict[value] if value in value_dict else str(value)
        for value in range(raster_values[0], raster_values[1] + 1)
    ]
    band.SetRasterCategoryNames(full_labels)

    # set band name
    band.SetDescription(raster_band_label)

    del band, raster


def make_scenario_suffix(file_suffix: str, scenario_suffix: str = None) -> str:
    """Create a suffix for output files based on the file_suffix and scenario_suffix
    in the following format: "_{file_suffix}_{scenario_suffix}"

    Args:
        file_suffix (str): suffix to append to output files
        scenario_suffix (str): suffix to append to output files

    Returns:
        str: suffix for output files
    """
    if file_suffix != "" and not file_suffix.startswith("_"):
        file_suffix = "_" + file_suffix
    if (
        scenario_suffix is not None
        and scenario_suffix != ""
        and not scenario_suffix.startswith("_")
        and not file_suffix.endswith("_")
    ):
        scenario_suffix = "_" + scenario_suffix

    return_suffix = (
        file_suffix if scenario_suffix is None else file_suffix + scenario_suffix
    )

    return return_suffix


def conditional_vector_project(
    vector_path: pathlike, projection_wkt: str, workspace_path: pathlike
) -> Path:
    """Tests if the selected vector is in the selected projection and if not, project it and create new file.

    Args:
        vector_path (pathlike): Path to vector to conditionally project
        projection_wkt (str): Well-Known_Text of desired geospatial projection
        workspace_path (pathlike): Path to folder in which to deposit resulting file, if projected.

    Returns:
        target_path (Path): Path to resulting vector (either the original or the newly projected)
    """

    # Ensure all pathlike variables are paths
    parcel_vector = Path(parcel_vector)
    workspace_path = Path(workspace_path)

    vector_info = pygeoprocessing.get_vector_info(str(vector_path))
    if vector_info["projection_wkt"] != projection_wkt:
        target_path = workspace_path / vector_path.name
        if target_path.is_file():
            target_path = (
                workspace_path / f"{vector_path.stem}_proj{vector_path.suffix}"
            )
        logger.debug(f"Reprojecting {vector_path.name}")
        pygeoprocessing.reproject_vector(
            str(vector_path), projection_wkt, str(target_path)
        )
    else:
        target_path = vector_path
    return target_path


def grouped_scalar_calculation(
    base_raster_path: pathlike,
    category_raster_path: pathlike,
    target_raster_path: pathlike,
    category_list: List[int],
    scalar_list: List[int],
) -> None:
    """Raster calculator that multiplies some base raster by scalars associated with a
    different raster's categories

    Args:
        base_raster_path (pathlike): Path to base raster to multiply
        category_raster_path (pathlike): Path to raster with categories
        target_raster_path (pathlike): Path to target raster
        category_list (List[int]): List of categories in category raster
        scalar_list (List[int]): List of scalars associated with categories

    Returns:
        None

    """
    # Ensure path variables are Path objects
    base_raster_path = Path(base_raster_path)
    category_raster_path = Path(category_raster_path)
    target_raster_path = Path(target_raster_path)

    temporary_working_dir = Path(tempfile.mkdtemp(dir=target_raster_path.parent))

    base_raster_info = pygeoprocessing.get_raster_info(str(base_raster_path))
    base_raster_nodata = base_raster_info["nodata"][0]
    cell_size = np.min(np.abs(base_raster_info["pixel_size"]))

    category_raster_nodata = pygeoprocessing.get_raster_info(str(category_raster_path))[
        "nodata"
    ][0]

    # Calculate kWh map
    grouped_scalar_op = rasterops.MultiplyRasterByScalarList(
        category_list, scalar_list, base_raster_nodata, category_raster_nodata
    )

    temp_base_path = temporary_working_dir / Path(base_raster_path).name
    temp_category_path = temporary_working_dir / Path(category_raster_path).name

    pygeoprocessing.align_and_resize_raster_stack(
        [str(base_raster_path), str(category_raster_path)],
        [str(temp_base_path), str(temp_category_path)],
        ["near", "near"],
        (cell_size, -cell_size),
        "intersection",
    )

    pygeoprocessing.raster_calculator(
        [(str(temp_base_path), 1), (str(temp_category_path), 1)],
        grouped_scalar_op,
        str(target_raster_path),
        gdal.GDT_Float32,
        base_raster_nodata,
    )


def extract_admin_intersections(
    study_area_path: Path, admin_boundaries_path: Path, admin_id_fields: dict
) -> dict:
    """Function that returns the names of admin boundaries that intersect the study area.

    Parameters:
        study_area_path (Path): pathlib Path to a shapefile of the study area.
        admin_boundaries_path (Path): pathlib Path to a shapefile of the administrative boundaries.
        admin_id_fields (dict): dict of field names that contains desired administrative identifiers.

    Returns:
        Dict of administrative identifiers for areas that intersected the study area.
    """

    # Read in study area and extract geometry
    driver = ogr.GetDriverByName("ESRI Shapefile")
    study_area = driver.Open(str(study_area_path), 0)
    study_area_layer = study_area.GetLayer()
    study_area_feat = study_area_layer.GetFeature(
        0
    )  # Assumes a single contiguous study area outline
    study_area_geom = study_area_feat.GetGeometryRef()

    # Read in admin boundaries
    admin_boundaries = driver.Open(str(admin_boundaries_path), 0)
    admin_layer = admin_boundaries.GetLayer()
    admin_layer.SetSpatialFilter(study_area_geom)

    # Output field admin field values
    output_admin_values = {key: list() for key in admin_id_fields.keys()}
    for feature in admin_layer:
        for id_type, field in admin_id_fields.items():
            output_admin_values[id_type].append(feature.GetField(field))

    # Reset readings and close files
    admin_layer.ResetReading()
    study_area_layer.ResetReading()
    admin_boundaries = None
    study_area = None
    study_area_feat = None
    study_area_geom = None

    return output_admin_values


def extract_monthly_temperatures(temp_data_dir: Path, aoi_vector_path: Path, year=2017):
    """Extract monthly average temperature data for a given year based on a pre-defined area of interest.

    Parameters:
        temp_data_dir (Path): Pathlib Path to a folder containing temperature data from SOURCE
        aoi_vector_path (Path): Pathlib Path to the AOI vector file
        year (int): the year in question, must be between 1900 and 2017

    Returns:
        A Series containing lon, lat, and average monthly temperature values.
    """

    # TODO Check on the source of the temperature data

    # Test if input year is within the correct range
    assert 1900 <= year <= 2017, f"Year {year} is not between 1900 and 2017"

    def min_dimension(dataframe, column):
        column_data = abs(dataframe[column] - dataframe[column].shift(1))
        column_data = column_data.dropna()
        distance = column_data[column_data != 0].min()
        return distance

    # Import gridded global average temperature data for a given year
    temp_df = pd.read_csv(
        temp_data_dir / f"air_temp.{year}",
        delim_whitespace=True,
        index_col=False,
        header=None,
        names=[
            "lon",
            "lat",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            "",
        ],
    ).iloc[:, :-1]
    temp_gdf = gpd.GeoDataFrame(
        temp_df, geometry=gpd.points_from_xy(temp_df["lon"], temp_df["lat"])
    ).set_crs(epsg=4326)
    aoi_gdf = gpd.read_file(aoi_vector_path).to_crs(temp_gdf.crs)
    # Add dummy column to dissolve all geometries into one
    aoi_gdf["dummy"] = "dummy"
    aoi_geom = aoi_gdf.dissolve(by="dummy").geometry.iloc[0]
    subset_gdf = temp_gdf[temp_gdf.within(aoi_geom)]

    # If within calculation doesn't work, buffer AOI by half of the space between point locations until it does
    buffer_dist = 0
    buffer_add = min(min_dimension(temp_gdf, "lon"), min_dimension(temp_gdf, "lat")) / 4
    while subset_gdf.empty:
        buffer_dist += buffer_add
        with warnings.catch_warnings():
            # this will suppress all warnings in this block
            warnings.simplefilter("ignore")

            buffer_geom = (
                aoi_gdf.dissolve(by="dummy").buffer(buffer_dist).geometry.iloc[0]
            )
            subset_gdf = temp_gdf[temp_gdf.within(buffer_geom)]

    # Calculate average temperatures
    temp_means = pd.DataFrame(
        subset_gdf.drop(columns=["geometry"]).mean()[2:], columns=["temperature"]
    )

    return temp_means
