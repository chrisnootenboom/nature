import logging
from pathlib import Path
import typing
from typing import List, Set, Dict, Tuple, Optional
import tempfile
import shutil

import requests
import xmltodict

import math
import pandas as pd
import numpy as np
import pint

import matplotlib.colors

import rasterio as rio
from rasterio.mask import mask
from osgeo import gdal, osr
import geopandas as gpd

import pygeoprocessing
import natcap.invest.utils
import natcap.invest.spec_utils
import natcap.invest.validation

from . import functions

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

pathlike = str | Path

# typing package for type hinting in functions


# TODO Go through and clean up NoData entries, since I would like to internalize that parameter as much as possible.
# NOTE The main thing is to inherit NoData from the input raster. If changing raster dtype, then check if nodata fits in new dtype.


def sjoin_representative_points(
    target_gdf,
    join_gdf_list,
    join_fields_list_list,
):
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

    logger.info(
        f"Starting a convolution over {signal_raster_path} with a "
        f"decay distance of {decay_kernel_distance}"
    )
    temporary_working_dir = Path(
        tempfile.mkdtemp(dir=target_convolve_raster_path.parents[0])
    )
    exponential_kernel_path = temporary_working_dir / "exponential_decay_kernel.tif"
    natcap.invest.utils.exponential_decay_kernel_raster(
        decay_kernel_distance, str(exponential_kernel_path)
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
):
    """Convolve signal by an logicstic decay of a given radius and start distance.

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

    logger.info(
        f"Starting a logistic convolution over {signal_raster_path} with a "
        f"decay starting at {decay_start_distance} and terminating at {terminal_kernel_distance}"
    )
    temporary_working_dir = Path(
        tempfile.mkdtemp(dir=target_convolve_raster_path.parents[0])
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

    # Get CDL data from NASS as XML link
    nass_xml_url = rf"https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year={year}&fips={fips}"
    r = requests.get(nass_xml_url)

    # Extract XML link from XML
    nass_data_url = xmltodict.parse(r.content)["ns1:GetCDLFileResponse"]["returnURL"]

    # Download CDL data to local file
    r = requests.get(nass_data_url, allow_redirects=True)
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
):
    """A function to copy a raster to a new datatype

    Args:
        raster_path (str or pathlib.Path): Path to the input raster
        output_raster_path (pathlib.Path): Path to the output raster
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


def batch_masked_zonal_stats(
    zonal_raster_list,
    mask_raster_path,
    mask_raster_value,
    zonal_vector_path,
    temp_workspace_dir,
    zonal_join_columns=[
        "count",
        "nodata_count",
        "sum",
        "mean",
    ],
) -> Tuple[List[pd.DataFrame], Path]:
    """A function to calculate masked zonal statistics for a list of rasters

    Args:
        static_sampling_raster_list (_type_): _description_
        mask_raster_path (_type_): _description_
        zonal_vector_path (_type_): _description_
        temp_workspace_dir (_type_): _description_
        zonal_join_columns (list, optional): _description_. Defaults to [ "count", "nodata_count", "sum", "mean", ].

    Returns:
        List[pd.DataFrame]: _description_
        Path: _description_
    """
    zonal_stats_list = []
    aligned_mask_raster_path = temp_workspace_dir / f"{mask_raster_path.stem}.tif"
    for i, raster in enumerate(zonal_raster_list):
        if i == 0:
            logger.info(f"Aligning mask with zonal raster {raster}")
            raster_info = pygeoprocessing.get_raster_info(str(raster))
            pygeoprocessing.align_and_resize_raster_stack(
                [str(raster), str(mask_raster_path)],
                [
                    str(temp_workspace_dir / f"{(raster).stem}.tif"),
                    str(aligned_mask_raster_path),
                ],
                ["near", "near"],
                raster_info["pixel_size"],
                "intersection",
                raster_align_index=0,
            )
        logger.info(f"Zonal statistics {i+1} of {len(zonal_raster_list)} | {raster}")
        zonal_stats_df = masked_zonal_stats(
            raster,
            aligned_mask_raster_path,
            mask_raster_value,
            zonal_vector_path,
            temp_workspace_dir,
        )

        zonal_stats_df = zonal_stats_df[zonal_join_columns]
        zonal_stats_df = zonal_stats_df.rename(
            columns={col: f"{(raster).stem}__{col}" for col in zonal_join_columns}
        )
        zonal_stats_list.append(zonal_stats_df)

    return zonal_stats_list, aligned_mask_raster_path


def masked_zonal_stats(
    base_raster_path,
    mask_raster_path,
    mask_value,
    zonal_vector_path,
    workspace_dir,
):
    """Calculate zonal statistics for a raster masked by a mask raster.

    Args:
        base_raster_path (_type_): _description_
        mask_raster_path (_type_): _description_
        mask_value (_type_): _description_
        zonal_vector_path (_type_): _description_
        workspace_dir (_type_): _description_

    Returns:
        _type_: _description_
    """

    base_raster_info = pygeoprocessing.get_raster_info(str(base_raster_path))
    base_nodata = base_raster_info["nodata"][0]

    def mask_op(base_array, mask_array):
        result = np.copy(base_array)
        result[mask_array != mask_value] = base_nodata
        return result

    target_mask_raster_path = workspace_dir / f"_masked_{base_raster_path.stem}.tif"

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


def rename_invest_results(invest_model, invest_args, suffix):
    """Rename InVEST results to include the suffix.

    Args:
        invest_model (str): InVEST model name.
        invest_info (dict): _description_
        suffix (str): String to append to the end of the file name.
    """

    # Modify relevant outputs
    # TODO Change pollination results to be dependent on pollinator species
    if invest_model == "pollination":
        result_file_list = [Path(invest_args["workspace_dir"]) / f"avg_pol_abd.tif"]
    elif invest_model == "pollination_mv":
        result_file_list = [
            Path(invest_args["workspace_dir"]) / f"marginal_value_general.tif"
        ]
    elif invest_model == "sdr":
        result_file_list = [
            Path(invest_args["workspace_dir"]) / f"sed_export.tif",
            Path(invest_args["workspace_dir"]) / f"watershed_results_sdr.shp",
        ]
    elif invest_model == "ndr":
        result_file_list = [
            Path(invest_args["workspace_dir"]) / f"n_total_export.tif",
            Path(invest_args["workspace_dir"]) / f"p_surface_export.tif",
            Path(invest_args["workspace_dir"]) / f"watershed_results_ndr.gpkg",
        ]
    elif invest_model == "carbon":
        result_file_list = [
            Path(invest_args["workspace_dir"]) / f"tot_c_cur.tif",
            Path(invest_args["workspace_dir"]) / f"tot_c_fut.tif",
            Path(invest_args["workspace_dir"]) / f"delta_cur_fut.tif",
        ]

    for result_file in result_file_list:
        if result_file.exists():
            result_file.rename(
                Path(invest_args["workspace_dir"])
                / f"{result_file.stem}_{suffix}{result_file.suffix}"
            )


def clip_raster_by_vector(mask_path, raster_path, outupt_raster_path):
    """Clip a raster by a vector mask.

    Args:
        mask_path (str): Path to the mask vector.
        raster_path (str): Path to the raster to be clipped.
        outupt_raster_path (str): Path to the output raster.
    """
    mask_gdf = gpd.read_file(mask_path)

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

    with rio.open(outupt_raster_path, "w", **out_meta) as dst:
        dst.write(out_image)


def burn_polygon_add(
    vector_path: pathlike,
    base_raster_path: pathlike,
    intermediate_workspace_path: pathlike,
    target_raster_path: pathlike,
    burn_value: int | float,
    rasterio_dtype=rio.int8,
    nodata_val: int | float = None,
    burn_attribute: str = None,
    constraining_raster_value_tuple: tuple = None,
):
    """Burns a polygon into a raster by adding a specified value.

    Args:
        vector_path (pathlike): Path to the vector to be burned.
        base_raster_path (pathlike): Path to the raster to be burned.
        intermediate_workspace_path (pathlike): Path to the intermediate workspace.
        target_raster_path (pathlike): Path to the output raster.
        burn_value (int): Value to be burned into the raster.
        rasterio_dtype (rasterio.dtype, optional): Rasterio datatype to use for the output raster.
        nodata_val (int/float, optional): Value to use for nodata. If None, will use the nodata value of the base raster.
        burn_attribute (str, optional): Attribute to use for burning. If None, will burn all polygons with the same value.

    Returns:
        None
    """

    # Make all path variables Path objects
    vector_path = Path(vector_path)
    base_raster_path = Path(base_raster_path)
    intermediate_workspace_path = Path(intermediate_workspace_path)

    # Dissolve vector to single feature to avoid overlapping polygons
    logger.info(f"Dissolving burn vector")
    vector_gdf = gpd.read_file(vector_path)
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

    # Burn and add vector to raster values
    logger.info(f"Burning vector into raster")
    if burn_attribute is not None:
        pygeoprocessing.rasterize(
            str(dissolved_vector_path),
            str(intermediate_raster_path),
            burn_values=[burn_value],
            option_list=["MERGE_ALG=ADD", f"ATTRIBUTE={burn_attribute}"],
        )
    else:
        pygeoprocessing.rasterize(
            str(dissolved_vector_path),
            str(intermediate_raster_path),
            burn_values=[burn_value],
            option_list=["MERGE_ALG=ADD"],
        )

    # Get raster info
    base_raster_info = pygeoprocessing.get_raster_info(str(base_raster_path))
    target_raster_info = pygeoprocessing.get_raster_info(str(intermediate_raster_path))

    # Constraining output by constraint raster, if called for
    if constraining_raster_value_tuple is not None:
        # Create constraining raster calculator function
        def _constrain_op(
            base_array, change_array, constraining_array, constraining_value
        ):
            out_array = np.where(
                constraining_array == constraining_value, change_array, base_array
            )

            return out_array

        # Replace values in intermediate raster with original raster values where constrained
        logger.info(f"Constraining burn by raster")
        constrained_raster_path = (
            intermediate_workspace_path
            / f"_constrained_burned_{Path(base_raster_path).stem}.tif"
        )
        pygeoprocessing.raster_calculator(
            [
                (str(base_raster_path), 1),
                (str(intermediate_raster_path), 1),
                (str(constraining_raster_value_tuple[0]), 1),
                (constraining_raster_value_tuple[1], "raw"),
            ],
            _constrain_op,
            str(constrained_raster_path),
            target_raster_info["datatype"],
            target_raster_info["nodata"][0],
        )

        # Update intermediate raster path for final processing step
        intermediate_raster_path = constrained_raster_path

    # Remove areas where vector burned over nodata
    logger.info(f"Removing nodata areas")

    def mask_op(base_array, mask_array, nodata):
        result = np.copy(base_array)
        result[mask_array == nodata] = nodata
        return result

    pygeoprocessing.raster_calculator(
        [
            (str(intermediate_raster_path), 1),
            (str(base_raster_path), 1),
            (base_raster_info["nodata"], "raw"),
        ],
        mask_op,
        str(target_raster_path),
        target_raster_info["datatype"],
        target_raster_info["nodata"][0],
    )


def overlay_on_top_of_lulc(
    vector_path,
    base_lulc_path,
    intermediate_workspace_path,
    target_raster_path,
    burn_value: int | float = 1,
    nodata_val: int | float = None,
    burn_attribute: str = None,
):
    """Burns a polygon into a raster by adding a specified value, ensuring that the burn values are using digits unused in the base raster.

    Args:
        vector_path (pathlike): Path to the vector to be burned.
        base_raster_path (pathlike): Path to the raster to be burned.
        intermediate_workspace_path (pathlike): Path to the intermediate workspace.
        target_raster_path (pathlike): Path to the output raster.
        burn_value (int): Value to be burned into the raster.
        nodata_val (int/float): Value to use for nodata. If None, will use the nodata value of the base raster.
        burn_attribute (str): Attribute to use for burning. If None, will burn all polygons with the same value.
    """

    # Extract maximum value of the raster to determine how many digits are needed
    with rio.open(base_lulc_path) as dataset:
        lulc_digits = len(str(int(dataset.statistics(1).max)))

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
        rasterio_dtype = rio.dtypes.get_minimum_dtype(vector_gdf[burn_attribute].max())
    else:
        intermediate_vector_path = vector_path
        rasterio_dtype = rio.dtypes.get_minimum_dtype(burn_value * 10**lulc_digits)

    burn_polygon_add(
        intermediate_vector_path,
        base_lulc_path,
        intermediate_workspace_path,
        target_raster_path,
        burn_value,
        rasterio_dtype,
        nodata_val,
        burn_attribute,
    )


def calculate_load_per_pixel(
    lulc_raster_path: pathlike,
    lucode_to_parameters: dict,
    target_load_raster: pathlike,
    nodata_val: float,
):
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
    category_labels: List[str],
    category_hex_colors: List[str],
    raster_band_label: str,
) -> None:
    """Bin raster into a smaller number of integer classes and assign labels and colors to each class.


    Args:
        raster_path (str): path to input raster
        output_raster_path (str): path to output raster
        raster_band_label (int): band number of input raster to use
        category_bins (list): list of bin edges
        category_labels (list): list of labels for each bin
        category_colors (list): list of colors for each bin

    """

    raster_type = rio.dtypes.get_minimum_dtype(len(category_labels) - 1)

    input_nodata_val = pygeoprocessing.get_raster_info(str(raster_path))["nodata"][0]

    # Check if input raster nodata value is within the range of the output raster type and adapt if necessary
    output_nodata_val = nodata_validation(raster_type, input_nodata_val)
    if output_nodata_val in range(len(category_labels)):
        logger.warning("Nodata value does not fit within raster value map.")

    # Create constraining raster calculator function
    def _bin_op(array, bins):
        """Bin and label rasters."""
        valid_mask = ~natcap.invest.utils.array_equals_nodata(array, input_nodata_val)
        result = np.empty_like(array)
        result[:] = output_nodata_val
        result[valid_mask] = np.array(pd.cut(array[valid_mask], bins, labels=False))
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
        list(range(len(category_labels))),
        category_labels,
        category_hex_colors,
        raster_band_label,
    )


def assign_raster_labels_and_colors(
    raster_path: pathlike,
    category_values: List[int],
    category_labels: List[str],
    category_hex_colors: List[str],
    raster_band_label: str,
):
    """Assign labels and colors to raster categories.

    Args:
        raster_path (str): path to input raster
        category_values (list): list of category values (must be in ascending order!)
        category_labels (list): list of labels for each category value
        category_colors (list): list of colors for each category value
        raster_band_label (int): band number of input raster to use

    """
    # TODO add validation error if category_values, category_labels, and category_hex_colors are not the same length
    # Read raster band
    raster = gdal.OpenEx(str(raster_path), gdal.GA_Update | gdal.OF_RASTER)
    band = raster.GetRasterBand(1)

    # # Raster Attribute Table
    # rat = gdal.RasterAttributeTable()

    # rat.CreateColumn("Value", gdal.GFT_Integer, gdal.GFU_MinMax)
    # rat.CreateColumn("Label", gdal.GFT_String, gdal.GFU_Name)

    # for i, label in enumerate(category_labels):
    #     rat.SetValueAsInt(i, 0, int(i))
    #     rat.SetValueAsString(i, 1, str(label))

    # band.SetDefaultRAT(rat)

    # GDAL Color Table
    color_table = gdal.ColorTable()

    for value, color in zip(category_values, category_hex_colors):
        color_table.SetColorEntry(
            int(value), tuple(int(c * 255) for c in matplotlib.colors.to_rgba(color))
        )

    # set color table and color interpretation
    band.SetRasterColorTable(color_table)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    # set category names
    value_dict = {
        value: label for value, label in zip(category_values, category_labels)
    }
    full_labels = [
        value_dict[value] if value in value_dict else str(value)
        for value in range(255 + 1)
    ]
    band.SetRasterCategoryNames(full_labels)

    # set band name
    band.SetDescription(raster_band_label)

    del band, raster


# def burn_polygon_add_constrained(
#     vector_path,
#     raster_path,
#     constraining_raster_path,
#     constraining_value,
#     intermediate_workspace_path,
#     output_raster_path,
#     burn_value,
#     target_nodata,
#     rasterio_dtype=rio.int16,
#     nodata_val: int | float = None,
#     burn_attribute: str = None,
#     coonstraining_raster_value_tuple: tuple = None,
# ):
#     """Burns a polygon into a raster by adding a specified value, but constrained by
#     another boolean raster.

#     Args:
#         vector_path (pathlike): Path to the vector to be burned.
#         base_raster_path (pathlike): Path to the raster to be burned.
#         constraining_raster_path (pathlike): Path to the raster to be used for constraining.
#         constraining_value (int): Value to be used for constraining.
#         intermediate_workspace_path (pathlike): Path to the intermediate workspace.
#         target_raster_path (pathlike): Path to the output raster.
#         burn_value (int): Value to be burned into the raster.
#         rasterio_dtype (rasterio.dtype): Rasterio datatype to use for the output raster.
#         nodata_val (int/float): Value to use for nodata. If None, will use the nodata value of the base raster.
#         burn_attribute (str): Attribute to use for burning. If None, will burn all polygons with the same value.

#     """

#     # Make all path variables Path objects
#     vector_path = Path(vector_path)
#     raster_path = Path(raster_path)
#     intermediate_workspace_path = Path(intermediate_workspace_path)

#     # Dissolve vector to single feature to avoid overlapping polygons
#     logger.info(f"Dissolving burn vector")
#     vector_gdf = gpd.read_file(vector_path)
#     vector_gdf = vector_gdf.dissolve()
#     dissolved_vector_path = (
#         intermediate_workspace_path / f"_dissolved_{vector_path.stem}.gpkg"
#     )
#     vector_gdf.to_file(dissolved_vector_path)

#     # Copy raster to different datatype to accommodate new integer values
#     intermediate_raster_path = (
#         intermediate_workspace_path / f"_burned_{Path(raster_path).stem}.tif"
#     )
#     if nodata_val is not None:
#         copy_raster_to_new_datatype(
#             raster_path,
#             intermediate_raster_path,
#             rasterio_dtype=rasterio_dtype,
#             nodata_val=nodata_val,
#         )
#     else:
#         copy_raster_to_new_datatype(
#             raster_path,
#             intermediate_raster_path,
#             rasterio_dtype=rasterio_dtype,
#         )

#     # Burn and add vector to raster values
#     logger.info(f"Burning vector into raster")
#     if burn_attribute is not None:
#         pygeoprocessing.rasterize(
#             str(dissolved_vector_path),
#             str(intermediate_raster_path),
#             burn_values=[burn_value],
#             option_list=["MERGE_ALG=ADD", f"ATTRIBUTE={burn_attribute}"],
#         )
#     else:
#         pygeoprocessing.rasterize(
#             str(dissolved_vector_path),
#             str(intermediate_raster_path),
#             burn_values=[burn_value],
#             option_list=["MERGE_ALG=ADD"],
#         )

#     # Create constraining raster calculator function
#     def _constrain_op(base_array, change_array, constraining_array, constraining_value):
#         out_array = np.where(
#             constraining_array == constraining_value, change_array, base_array
#         )

#         return out_array

#     # Replace values in intermediate raster with original raster values where constrained
#     logger.info(f"Constraining burn by raster")
#     pygeoprocessing.raster_calculator(
#         [
#             (str(raster_path), 1),
#             (str(intermediate_raster_path), 1),
#             (str(constraining_raster_path), 1),
#             (constraining_value, "raw"),
#         ],
#         _constrain_op,
#         str(output_raster_path),
#         gdal.GDT_Int16,
#         target_nodata,
#     )
