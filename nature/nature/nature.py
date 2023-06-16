import logging
from pathlib import Path

import requests
import xmltodict

import pandas as pd
import numpy as np

import rasterio as rio
from rasterio.mask import mask
from osgeo import gdal
import pygeoprocessing
import geopandas as gpd

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def extract_cdl(year, fips, temp_dir):
    """Extract CDL data for a given year and FIPS code

    Args:
        year (int): year to extract
        fips (int): FIPS code to extract
        temp_dir (pathlib.Path): temporary directory to store data

    Returns:
        pathlib.Path: path to extracted CDL data
    """
    logger.info(f"Extracting CDL data for {year} in FIPS {fips}")
    # Get CDL data from NASS as XML link
    nass_xml_url = rf"https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year={year}&fips={fips}"
    r = requests.get(nass_xml_url)
    # Extract XML link from XML
    nass_data_url = xmltodict.parse(r.content)["ns1:GetCDLFileResponse"]["returnURL"]
    # Download CDL data
    r = requests.get(nass_data_url, allow_redirects=True)
    outfile = temp_dir / f"CDL_{year}_{fips}.tif"
    with open(outfile, "wb") as cdl_out:
        cdl_out.write(r.content)

    return outfile


def copy_raster_to_new_datatype(
    raster_path, output_raster_path, rasterio_dtype=rio.int16
):
    """A function to copy a raster to a new datatype

    Args:
        raster_path (pathlib.Path): Path to the input raster
        output_raster_path (pathlib.Path): Path to the output raster
    """
    src = rio.open(raster_path)
    src_array = src.read(1)
    profile = src.profile
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
):
    """A function to calculate masked zonal statistics for a list of rasters

    Args:
        static_sampling_raster_list (_type_): _description_
        mask_raster_path (_type_): _description_
        zonal_vector_path (_type_): _description_
        temp_workspace_dir (_type_): _description_
        zonal_join_columns (list, optional): _description_. Defaults to [ "count", "nodata_count", "sum", "mean", ].

    Returns:
        _type_: _description_
        _type_: _description_
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


def rename_invest_results(invest_model, invest_info, suffix):
    """Rename InVEST results to include the suffix.

    Args:
        invest_model (str): InVEST model name.
        invest_info (dict): _description_
        suffix (str): String to append to the end of the file name.
    """

    # Modify relevant outputs
    if invest_model == "pollination":
        result_file_list = [
            Path(invest_info["args"]["workspace_dir"]) / f"avg_pol_abd.tif"
        ]
    elif invest_model == "pollination_mv":
        result_file_list = [
            Path(invest_info["args"]["workspace_dir"]) / f"marginal_value_general.tif"
        ]
    elif invest_model == "sdr":
        result_file_list = [
            Path(invest_info["args"]["workspace_dir"]) / f"sed_export.tif",
            Path(invest_info["args"]["workspace_dir"]) / f"watershed_results_sdr.shp",
        ]
    elif invest_model == "ndr":
        result_file_list = [
            Path(invest_info["args"]["workspace_dir"]) / f"n_total_export.tif",
            Path(invest_info["args"]["workspace_dir"]) / f"p_surface_export.tif",
            Path(invest_info["args"]["workspace_dir"]) / f"watershed_results_ndr.gpkg",
        ]
    elif invest_model == "carbon":
        result_file_list = [
            Path(invest_info["args"]["workspace_dir"]) / f"tot_c_cur.tif",
            Path(invest_info["args"]["workspace_dir"]) / f"tot_c_fut.tif",
            Path(invest_info["args"]["workspace_dir"]) / f"delta_cur_fut.tif",
        ]

    for result_file in result_file_list:
        if result_file.exists():
            result_file.rename(
                Path(invest_info["args"]["workspace_dir"])
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
    vector_path, raster_path, output_raster_path, burn_value, rasterio_dtype=rio.int16
):
    """Burns a polygon into a raster by adding a specified value.

    Args:
        vector_path (_type_): _description_
        raster_path (_type_): _description_
        output_raster_path (_type_): _description_
        burn_value (_type_): _description_
    """
    copy_raster_to_new_datatype(
        raster_path, output_raster_path, rasterio_dtype=rasterio_dtype
    )
    pygeoprocessing.rasterize(
        str(vector_path),
        str(output_raster_path),
        burn_values=[burn_value],
        option_list=["MERGE_ALG=ADD"],
    )


def burn_polygon_add_constrained(
    vector_path,
    raster_path,
    constraining_raster_path,
    constraining_value,
    intermediate_workspace_path,
    output_raster_path,
    burn_value,
    target_nodata,
    rasterio_dtype=rio.int16,
):
    """Burns a polygon into a raster by adding a specified value, but constrained by
    another boolean raster.

    Args:
        vector_path (_type_): _description_
        raster_path (_type_): _description_
        constraining_raster_path (_type_): _description_
        constraining_value (_type_): _description_
        intermediate_workspace_path (_type_): _description_
        output_raster_path (_type_): _description_
        burn_value (_type_): _description_
        target_nodata (_type_): _description_
    """
    # Make all path variables Path objects
    vector_path = Path(vector_path)
    raster_path = Path(raster_path)
    intermediate_workspace_path = Path(intermediate_workspace_path)

    # Dissolve vector to single feature to avoid overlapping polygons
    vector_gdf = gpd.read_file(vector_path)
    vector_gdf = vector_gdf.dissolve()
    dissolved_vector_path = (
        intermediate_workspace_path / f"_dissolved_{vector_path.stem}.gpkg"
    )
    vector_gdf.to_file(dissolved_vector_path)
    # Copy raster to int16 to allow for larger integer values
    intermediate_raster_path = (
        Path(intermediate_workspace_path) / f"_solar_{Path(raster_path).stem}.tif"
    )
    copy_raster_to_new_datatype(
        raster_path, intermediate_raster_path, rasterio_dtype=rasterio_dtype
    )
    # Burn and add vector to raster values
    pygeoprocessing.rasterize(
        str(dissolved_vector_path),
        str(intermediate_raster_path),
        burn_values=[burn_value],
        option_list=["MERGE_ALG=ADD"],
    )

    # Create constraining raster calculator function
    def _constrain_op(base_array, change_array, constraining_array, constraining_value):
        out_array = np.where(
            constraining_array == constraining_value, change_array, base_array
        )

        return out_array

    # Replace values in intermediate raster with original raster values where constrained
    pygeoprocessing.raster_calculator(
        [
            (str(raster_path), 1),
            (str(intermediate_raster_path), 1),
            (str(constraining_raster_path), 1),
            (constraining_value, "raw"),
        ],
        _constrain_op,
        str(output_raster_path),
        gdal.GDT_Int16,
        target_nodata,
    )
