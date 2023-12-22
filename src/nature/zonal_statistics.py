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
import rasterstats

import pygeoprocessing
from pygeoprocessing.geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
import pygeoprocessing.kernels
import natcap.invest.utils
import natcap.invest.spec_utils
import natcap.invest.validation

from . import functions
from . import rasterops

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

pathlike = str | Path


def zonal_stats(
    zonal_raster_path: pathlike,
    zonal_vector_path: pathlike,
    zonal_join_columns: list = [
        "count",
        "nodata_count",
        "sum",
        "mean",
    ],
    zonal_raster_label: str = None,
    join_to_vector: bool = False,
    output_vector_path: pathlike = None,
) -> pd.DataFrame | gpd.GeoDataFrame | None:
    """A function to calculate zonal statistics for a raster and, if desired, join to the zonal vector

    Args:
        zonal_raster_path (pathlike): Path to raster to calculate zonal statistics for
        zonal_vector_path (pathlike): Path to vector to calculate zonal statistics for
        zonal_join_columns (list): List of zonal statistics to calculate
        zonal_raster_label (str): Label for zonal raster
        join_to_vector (bool): Whether to join zonal statistics to vector
        output_vector_path (pathlike): Optional, Path to output vector

    Returns:
        pandas.DataFrame: Zonal statistics dataframe (if join_to_vector == False)
        gpd.GeoDataFrame: Joined zonal statistics dataframe (if join_to_vector == True and output_vector_path not provided)
        None: None (if join_to_vector == True and output_vector_path provided)
    """
    zonal_raster_path = Path(zonal_raster_path)
    with warnings.catch_warnings():
        # this will suppress all warnings in this block
        zonal_stats_dict = pygeoprocessing.zonal_statistics(
            (str(zonal_raster_path), 1), str(zonal_vector_path)
        )
    zonal_stats_df = pd.DataFrame(zonal_stats_dict).transpose()

    zonal_stats_df["mean"] = zonal_stats_df["sum"] / zonal_stats_df["count"]

    zonal_stats_df = zonal_stats_df[zonal_join_columns]
    # Rename columns to labels if provided, else raster name
    if zonal_raster_label is not None:
        zonal_stats_df = zonal_stats_df.rename(
            columns={col: f"{zonal_raster_label}__{col}" for col in zonal_join_columns}
        )
    else:
        zonal_stats_df = zonal_stats_df.rename(
            columns={
                col: f"{(zonal_raster_path).stem}__{col}" for col in zonal_join_columns
            }
        )

    if join_to_vector:
        # Join all zonal statistics results together
        results_gdf = gpd.read_file(
            zonal_vector_path, engine="pyogrio", fid_as_index=True
        )
        results_gdf = results_gdf.merge(
            zonal_stats_df,
            left_index=True,
            right_index=True,
        )
        if output_vector_path is not None:
            results_gdf.to_file(output_vector_path, driver="GPKG")
            return None
        else:
            return results_gdf
    else:
        return zonal_stats_df


def batch_zonal_stats(
    zonal_raster_list: list,
    zonal_vector_path: pathlike,
    temp_workspace_path: pathlike,
    zonal_join_columns: list = [
        "count",
        "nodata_count",
        "sum",
        "mean",
    ],
    zonal_raster_labels: List[str] = None,
    join_to_vector: bool = False,
    output_vector_path: pathlike = None,
) -> List[pd.DataFrame] | gpd.GeoDataFrame | None:
    """A function to calculate zonal statistics for a list of rasters and, if desired, join to the zonal vector

    Args:
        zonal_raster_list (list): List of paths to rasters to calculate zonal statistics for
        zonal_vector_path (pathlike): Path to vector to calculate zonal statistics for
        temp_workspace_path (pathlike): Path to temporary workspace
        zonal_join_columns (list): List of zonal statistics to calculate
        zonal_raster_labels (list): List of labels for zonal rasters
        output_vector_path (pathlike): Optional, Path to output vector

    Returns:
        List: List of zonal statistics dataframes (if join_to_vector == False)
        gpd.GeoDataFrame: Joined zonal statistics dataframe (if join_to_vector == True and output_vector_path not provided)
        None: None (if join_to_vector == True and output_vector_path provided)
    """

    if zonal_raster_labels is not None:
        assert len(zonal_raster_labels) == len(zonal_raster_list), (
            f"Length of zonal_raster_labels ({len(zonal_raster_labels)}) must match "
            f"length of zonal_raster_list ({len(zonal_raster_list)})."
        )

    # Ensure path arguments are Path objects
    zonal_vector_path = Path(zonal_vector_path)
    temp_workspace_path = Path(temp_workspace_path)

    zonal_stats_df_list = []
    for i, raster in enumerate(zonal_raster_list):
        raster = Path(raster)
        logger.debug(f"Zonal statistics {i+1} of {len(zonal_raster_list)} | {raster}")
        zonal_stats_df = zonal_stats(
            raster,
            zonal_vector_path,
            zonal_join_columns=zonal_join_columns,
            zonal_raster_label=zonal_raster_labels[i],
        )

        zonal_stats_df_list.append(zonal_stats_df)

    if join_to_vector:
        # Join all zonal statistics results together
        zonal_stats_df = pd.concat(zonal_stats_df_list, axis=1)

        results_gdf = gpd.read_file(
            zonal_vector_path, engine="pyogrio", fid_as_index=True
        )
        results_gdf = results_gdf.merge(
            zonal_stats_df,
            left_index=True,
            right_index=True,
        )
        if output_vector_path is not None:
            results_gdf.to_file(output_vector_path, driver="GPKG")
            return None
        else:
            return results_gdf
    else:
        return zonal_stats_df_list


def pickle_zonal_stats(
    base_vector_path: pathlike,
    base_raster_path: pathlike,
    target_pickle_path: pathlike,
    zonal_join_columns: list = [
        "count",
        "nodata_count",
        "sum",
        "mean",
    ],
    label: str = None,
) -> None:
    """Calculate Zonal Stats for a vector/raster pair and pickle result.

    Args:
        base_vector_path (pathlike): path to vector file
        base_raster_path (pathlike): path to raster file to aggregate over.
        target_pickle_path (pathlike): path to desired target pickle file that will
            be a pickle of the pygeoprocessing.zonal_stats function.

    Returns:
        None.

    """
    # Ensure path arguments are Path objects
    base_vector_path = Path(base_vector_path)
    base_raster_path = Path(base_raster_path)
    target_pickle_path = Path(target_pickle_path)

    logger.info(
        f"Taking zonal statistics of {str(base_vector_path)} "
        f"over {str(base_raster_path)}"
    )
    zonal_stats_dict = pygeoprocessing.zonal_statistics(
        (str(base_raster_path), 1), str(base_vector_path), polygons_might_overlap=True
    )
    zonal_stats_df = pd.DataFrame(zonal_stats_dict).transpose()
    zonal_stats_df.index += -1  # Make sure indices match

    zonal_stats_df["mean"] = zonal_stats_df["sum"] / zonal_stats_df["count"]

    zonal_stats_df = zonal_stats_df[zonal_join_columns]
    # Rename columns to labels if provided, else raster name
    if label is not None:
        zonal_stats_df = zonal_stats_df.rename(
            columns={col: f"{label}__{col}" for col in zonal_join_columns}
        )
    else:
        zonal_stats_df = zonal_stats_df.rename(
            columns={
                col: f"{(base_raster_path).stem}__{col}" for col in zonal_join_columns
            }
        )

    with open(target_pickle_path, "wb") as pickle_file:
        pickle.dump(zonal_stats_df, pickle_file)


def batch_pickle_zonal_stats(
    zonal_raster_list: list,
    zonal_vector_path: pathlike,
    pickle_path_list: list,
    zonal_join_columns_list: List[list] = None,
    zonal_raster_labels: List[str] = None,
) -> Tuple[List[pd.DataFrame], Path]:
    """A function to calculate and pickle zonal statistics for a list of rasters

    Args:
        zonal_raster_list (list): List of paths to rasters to calculate zonal statistics for
        zonal_vector_path (pathlike): Path to vector to calculate zonal statistics for
        pickle_path_list (list): List of paths to pickle zonal statistics dataframes to
        zonal_join_columns (list): List of zonal statistics to calculate
        zonal_raster_labels (list): List of labels for zonal rasters

    Returns:
        tuple: Tuple of list of zonal statistics dataframes and path to aligned mask raster
    """

    assert len(pickle_path_list) == len(zonal_raster_list), (
        f"Length of pickle_path_list ({len(pickle_path_list)}) must match "
        f"length of zonal_raster_list ({len(zonal_raster_list)})."
    )
    if zonal_raster_labels is not None:
        assert len(zonal_raster_labels) == len(zonal_raster_list), (
            f"Length of zonal_raster_labels ({len(zonal_raster_labels)}) must match "
            f"length of zonal_raster_list ({len(zonal_raster_list)})."
        )
    if zonal_join_columns_list is not None:
        assert len(zonal_join_columns_list) == len(zonal_raster_list), (
            f"Length of zonal_join_columns_list ({len(zonal_join_columns_list)}) must match "
            f"length of zonal_raster_list ({len(zonal_raster_list)})."
        )

    # Ensure path arguments are Path objects
    zonal_vector_path = Path(zonal_vector_path)

    if zonal_join_columns_list is None:
        zonal_join_columns_list = [
            [
                "count",
                "nodata_count",
                "sum",
                "mean",
            ]
        ] * len(zonal_raster_list)

    if zonal_raster_labels is None:
        zonal_raster_labels = [None] * len(zonal_raster_list)

    for zonal_raster_path, pickle_path, columns, label in zip(
        zonal_raster_list,
        pickle_path_list,
        zonal_join_columns_list,
        zonal_raster_labels,
    ):
        pickle_zonal_stats(
            zonal_vector_path,
            zonal_raster_path,
            pickle_path,
            zonal_join_columns=columns,
            label=label,
        )


def join_batch_pickle_zonal_stats(
    pickle_file_path_list: List[pathlike],
    zonal_vector_path: pathlike,
    output_vector_path: pathlike = None,
):
    """A function to join a list of pickled zonal statistics dataframes to a vector

    Args:
        pickle_file_list (list): List of paths to pickled zonal statistics dataframes
        zonal_vector_path (pathlike): Path to vector to join to
        output_vector_path (pathlike): Optional, Path to output vector

    Returns:
        gpd.GeoDataFrame | None
    """
    # Ensure path arguments are Path objects
    zonal_vector_path = Path(zonal_vector_path)

    zonal_stats_df_list = []
    for pickle_file_path in pickle_file_path_list:
        with open(pickle_file_path, "rb") as pickle_file:
            zonal_stats_df_list.append(pickle.load(pickle_file))

    # Join all zonal statistics results together
    zonal_stats_df = pd.concat(zonal_stats_df_list, axis=1)

    results_gdf = gpd.read_file(zonal_vector_path, engine="pyogrio", fid_as_index=True)
    results_gdf = results_gdf.merge(
        zonal_stats_df,
        left_index=True,
        right_index=True,
    )
    if output_vector_path is not None:
        results_gdf.to_file(output_vector_path, driver="GPKG")
        return None
    else:
        return results_gdf


def masked_zonal_stats(
    base_raster_path: pathlike,
    mask_raster_path: pathlike,
    mask_value: int | float,
    zonal_vector_path: pathlike,
    workspace_path: pathlike,
) -> pd.DataFrame:
    """Calculate zonal statistics for a raster masked by a mask raster.

    Args:
        base_raster_path (pathlike): path to raster file to aggregate over.
        mask_raster_path (pathlike): path to raster file to use as mask.
        mask_value (int or float): value to use as mask.
        zonal_vector_path (pathlike): path to vector file to aggregate over.
        workspace_path (pathlike): path to desired workspace directory.

    Returns:
        pandas.DataFrame: dataframe of zonal statistics.
    """
    # Ensure path arguments are Path objects
    base_raster_path = Path(base_raster_path)
    mask_raster_path = Path(mask_raster_path)
    zonal_vector_path = Path(zonal_vector_path)
    workspace_path = Path(workspace_path)

    base_raster_info = pygeoprocessing.get_raster_info(str(base_raster_path))
    base_nodata = base_raster_info["nodata"][0]

    def mask_op(base_array, mask_array):
        result = np.copy(base_array)
        result[mask_array != mask_value] = base_nodata
        return result

    target_mask_raster_path = workspace_path / f"_masked_{base_raster_path.stem}.tif"

    logger.debug("Masking raster")
    pygeoprocessing.raster_calculator(
        [(str(base_raster_path), 1), (str(mask_raster_path), 1)],
        mask_op,
        str(target_mask_raster_path),
        base_raster_info["datatype"],
        base_nodata,
    )

    logger.debug("Calculating zonal statistics")
    zonal_stats_dict = pygeoprocessing.zonal_statistics(
        (str(target_mask_raster_path), 1), str(zonal_vector_path)
    )
    zonal_stats_df = pd.DataFrame(zonal_stats_dict).transpose()
    zonal_stats_df.index += -1  # Make sure indices match

    zonal_stats_df["mean"] = zonal_stats_df["sum"] / zonal_stats_df["count"]
    # os.remove(target_mask_raster_path)

    return zonal_stats_df


def batch_masked_zonal_stats(
    zonal_raster_list: list,
    mask_raster_path: pathlike,
    mask_raster_value: int | float,
    zonal_vector_path: pathlike,
    temp_workspace_path: pathlike,
    zonal_join_columns: list = [
        "count",
        "nodata_count",
        "sum",
        "mean",
    ],
    zonal_raster_labels: List[str] = None,
    join_to_vector: bool = False,
    output_vector_path: pathlike = None,
) -> Tuple[List[pd.DataFrame], Path]:
    """A function to calculate masked zonal statistics for a list of rasters

    Args:
        zonal_raster_list (list): List of paths to rasters to calculate zonal statistics for
        mask_raster_path (pathlike): Path to raster to use as mask
        mask_raster_value (int or float): Value to use as mask
        zonal_vector_path (pathlike): Path to vector to calculate zonal statistics for
        temp_workspace_path (pathlike): Path to temporary workspace
        zonal_join_columns (list): List of zonal statistics to calculate
        zonal_raster_labels (list): List of labels for zonal rasters

    Returns:
        tuple: Tuple of list of zonal statistics dataframes and path to aligned mask raster
    """

    if zonal_raster_labels is not None:
        assert len(zonal_raster_labels) == len(zonal_raster_list), (
            f"Length of zonal_raster_labels ({len(zonal_raster_labels)}) must match "
            f"length of zonal_raster_list ({len(zonal_raster_list)})."
        )

    # Ensure path arguments are Path objects
    mask_raster_path = Path(mask_raster_path)
    zonal_vector_path = Path(zonal_vector_path)
    temp_workspace_path = Path(temp_workspace_path)

    zonal_stats_df_list = []
    aligned_mask_raster_path = temp_workspace_path / f"{mask_raster_path.stem}.tif"
    for i, raster in enumerate(zonal_raster_list):
        raster = Path(raster)
        if i == 0 and not aligned_mask_raster_path.exists():
            logger.debug(f"Aligning mask with zonal raster {raster}")
            raster_info = pygeoprocessing.get_raster_info(str(raster))
            pygeoprocessing.align_and_resize_raster_stack(
                [str(raster), str(mask_raster_path)],
                [
                    str(temp_workspace_path / f"{(raster).stem}.tif"),
                    str(aligned_mask_raster_path),
                ],
                ["near", "near"],
                raster_info["pixel_size"],
                "intersection",
                raster_align_index=0,
            )
        logger.debug(f"Zonal statistics {i+1} of {len(zonal_raster_list)} | {raster}")

        zonal_stats_df = masked_zonal_stats(
            raster,
            aligned_mask_raster_path,
            mask_raster_value,
            zonal_vector_path,
            temp_workspace_path,
        )

        zonal_stats_df = zonal_stats_df[zonal_join_columns]
        # Rename columns to labels if provided, else raster name
        if zonal_raster_labels is not None:
            zonal_stats_df = zonal_stats_df.rename(
                columns={
                    col: f"{zonal_raster_labels[i]}_masked__{col}"
                    for col in zonal_join_columns
                }
            )
        else:
            zonal_stats_df = zonal_stats_df.rename(
                columns={
                    col: f"{(raster).stem}_masked__{col}" for col in zonal_join_columns
                }
            )
        zonal_stats_df_list.append(zonal_stats_df)

    if join_to_vector:
        # Join all zonal statistics results together
        zonal_stats_df = pd.concat(zonal_stats_df_list, axis=1)

        results_gdf = gpd.read_file(
            zonal_vector_path, engine="pyogrio", fid_as_index=True
        )
        results_gdf = results_gdf.merge(
            zonal_stats_df,
            left_index=True,
            right_index=True,
        )
        if output_vector_path is not None:
            results_gdf.to_file(output_vector_path, driver="GPKG")
            return aligned_mask_raster_path
        else:
            return results_gdf, aligned_mask_raster_path
    else:
        return zonal_stats_df_list, aligned_mask_raster_path


def categorize_vector_field_by_raster_zonal_stats(
    vector_path: pathlike,
    raster_path: pathlike,
    output_field: str,
    output_vector_path: pathlike,
    null_value: int | str,
    resample_size: int = None,
) -> None:
    """A function to calculate zonal stats for a vector/raster pair and assign
    the majority raster value to a new field in the vector.

    Args:
        vector_path (pathlike): path to vector file
        raster_path (pathlike): path to raster file to aggregate over.
        output_field (str): name of new field to create in vector
        output_vector_path (pathlike): path to desired output vector file
        null_value (int or str): value to assign to null values
        resample_size (int): size to resample raster to before zonal stats

    Returns:
        None
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        if resample_size is not None:
            logger.debug(f"Resampling raster to {resample_size}x{resample_size} pixels")

            zonal_raster_path = (
                Path(temp_dir)
                / f"{Path(raster_path).stem}_resampled_{resample_size}.tif"
            )
            pygeoprocessing.warp_raster(
                str(raster_path),
                (resample_size, resample_size),
                str(zonal_raster_path),
                "near",
            )
        else:
            zonal_raster_path = raster_path

        logger.debug(f"Performing zonal stats to calculate majority type")
        zonal_stats_list = rasterstats.zonal_stats(
            vector_path, zonal_raster_path, stats="majority"
        )

        zonal_stats_df = pd.DataFrame(zonal_stats_list)
        zonal_stats_df["majority"].fillna(null_value, inplace=True)
        zonal_stats_df.rename(columns={"majority": output_field}, inplace=True)

        logger.debug(f"Writing output raster to {output_vector_path}")
        vector_gdf = gpd.read_file(vector_path)
        vector_gdf = vector_gdf.join(zonal_stats_df)
        vector_gdf.to_file(output_vector_path, driver="GPKG")
