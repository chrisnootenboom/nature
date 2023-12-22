from unicodedata import name
import geopandas as gpd
from pathlib import Path
import os
import logging
import pygeoprocessing
import numpy as np
from osgeo import gdal, osr
import natcap.invest.utils
import natcap.invest.spec_utils

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


ARGS_SPEC = {
    "args": {
        "workspace_dir": natcap.invest.spec_utils.WORKSPACE,
        "results_suffix": natcap.invest.spec_utils.SUFFIX,
        "n_workers": natcap.invest.spec_utils.N_WORKERS,
        "population_raster_path": {},
        "ndvi_baseline_raster_path": {},
        "ndvi_scenario_raster_paths": {"type": list},
        "scenario_suffixes": {"type": list},
        "impact_distance_m": {"type": "number"},
        "mean_who5": {"type": "number"},
        "baseline_ndvi": {"type": "number"},
        "ndvi_increment": {"type": "number"},
        "per_capita_expenditures": {"type": "number"},
    }
}


NODATA_VAL = -9999


def get_raster_pixel_size(raster_path):
    raster_info = pygeoprocessing.get_raster_info(str(raster_path))
    pixel_size_tuple = raster_info["pixel_size"]
    try:
        mean_pixel_size = natcap.invest.utils.mean_pixel_size_and_area(
            pixel_size_tuple
        )[0]
    except ValueError:
        mean_pixel_size = np.min(np.absolute(pixel_size_tuple))
        logger.debug(
            f"Raster has unequal x, y pixel sizes: {pixel_size_tuple}. Using {mean_pixel_size} as the mean pixel size."
        )

    return mean_pixel_size


def average_kernel_raster(expected_distance, kernel_filepath):
    """Create a raster-based average kernel.

    The raster created will be a tiled GeoTiff, with 256x256 memory blocks.

    Args:
        expected_distance (int or float): The distance (in pixels) of the
            kernel's radius.
        kernel_filepath (string): The path to the file on disk where this
            kernel should be stored.  If this file exists, it will be
            overwritten.

    Returns:
        None
    """
    kernel_size = int(np.round(expected_distance * 2 + 1))

    driver = gdal.GetDriverByName("GTiff")
    kernel_dataset = driver.Create(
        kernel_filepath.encode("utf-8"),
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

    fill_array = np.empty((kernel_size, kernel_size))
    fill_array[:] = 1
    kernel_band.WriteArray(fill_array)

    kernel_band.FlushCache()
    kernel_dataset.FlushCache()
    kernel_band = None
    kernel_dataset = None


def execute(args):
    intermediate_output_dir = os.path.join(
        args["workspace_dir"], "intermediate_outputs"
    )
    try:
        os.makedirs(intermediate_output_dir)
    except OSError:
        if not os.path.isdir(intermediate_output_dir):
            raise

    # TODO Project data if necessary

    # TODO Removing negative NDVI values (as in Liu et al., 2019)

    # # Buffering the polygon by the impact distance (and double that)
    # # Changes in the wetland can affect neighborhoods up to a certain distance outside of the wetland, and neighborhoods themselves are affected by NDVI beyond that.
    # aoi_gdf = gpd.read_file(aoi_shp)
    # aoi_buffer_gdf = aoi_gdf.buffer(float(impact_distance_m))
    # aoi_buffer2_gdf = aoi_gdf.buffer(float(impact_distance_m * 2))

    # # TODO Remove scenario area from population?
    # buffer2_gdf = aoi_buffer2_gdf.difference(aoi_gdf)

    # Resample rasters
    logger.info(f"Aligning NDVI rasters")
    ndvi_scenario_rasters = args["ndvi_scenario_raster_paths"]
    ndvi_scenario_rasters.insert(0, args["ndvi_baseline_raster_path"])
    scenario_labels = args["scenario_suffixes"]
    scenario_labels.insert(0, "baseline")
    ndvi_scenario_resampled_rasters = [
        os.path.join(intermediate_output_dir, f"ndvi_{label}_resampled.tif")
        for label in scenario_labels
    ]

    ndvi_raster_info = pygeoprocessing.get_raster_info(
        args["ndvi_baseline_raster_path"]
    )
    ndvi_pixel_size_tuple = ndvi_raster_info["pixel_size"]

    pygeoprocessing.align_and_resize_raster_stack(
        [raster for raster in ndvi_scenario_rasters],
        [raster for raster in ndvi_scenario_resampled_rasters],
        ["near"] * (len(ndvi_scenario_rasters)),
        ndvi_pixel_size_tuple,
        "intersection",
    )

    # Create kernel for convolution
    ndvi_pixel_size = get_raster_pixel_size(args["ndvi_baseline_raster_path"])

    kernel_filepath = os.path.join(intermediate_output_dir, "average_kernel.tif")
    average_kernel_raster(
        int(args["impact_distance_m"] / ndvi_pixel_size), kernel_filepath
    )

    # Convolve all NDVI rasters
    ndvi_resample_input_rasters = []
    ndvi_resample_output_rasters = []
    for ndvi_raster, label in zip(ndvi_scenario_resampled_rasters, scenario_labels):
        logger.info(f"Convolving {label} NDVI raster")
        ndvi_average_tif = os.path.join(
            intermediate_output_dir, f"ndvi_average_{label}.tif"
        )
        resample_tif = os.path.join(args["workspace_dir"], f"ndvi_average_{label}.tif")

        pygeoprocessing.convolve_2d(
            (ndvi_raster, 1),
            (kernel_filepath, 1),
            ndvi_average_tif,
            ignore_nodata_and_edges=True,
            mask_nodata=True,
            normalize_kernel=True,
            target_nodata=NODATA_VAL,
        )

        ndvi_resample_input_rasters.append(ndvi_average_tif)
        ndvi_resample_output_rasters.append(resample_tif)

    # Resample rasters
    logger.info(f"Resampling NDVI rasters to population raster")
    population_resample_tif = os.path.join(
        intermediate_output_dir, "population_resampled.tif"
    )

    population_raster_info = pygeoprocessing.get_raster_info(
        args["population_raster_path"]
    )
    population_pixel_size_tuple = population_raster_info["pixel_size"]

    pygeoprocessing.align_and_resize_raster_stack(
        [args["population_raster_path"]]
        + [raster for raster in ndvi_resample_input_rasters],
        [population_resample_tif] + [raster for raster in ndvi_resample_output_rasters],
        ["near"] * (len(ndvi_resample_input_rasters) + 1),
        population_pixel_size_tuple,
        "intersection",
        raster_align_index=0,
    )

    # Calculate mental health (WHO-5 score) using function from Liu et al. (2019)
    who5_output_rasters = []
    for resample_ndvi, label in zip(ndvi_resample_output_rasters, scenario_labels):
        logger.info(f"Calculating WHO5 for {label} scenario")
        who5_raster = os.path.join(args["workspace_dir"], f"who5_{label}.tif")

        ndvi_nodata = pygeoprocessing.get_raster_info(resample_ndvi)["nodata"][0]

        def who5_op(ndvi, mean_who5, baseline_ndvi, ndvi_increment):
            """Raster calculator operation that to apply the WHO5 calculation.

            Args:
                ndvi (str): Scenario NDVI raster
                mean_who5 (float): Average WHO5 value prior to NDVI calculation
                baseline_ndvi (float): Baseline NDVI value for regression
                ndvi_increment (float): Per unit increase in WHO5 wellbeing.

            Returns:
                _type_: _description_
            """

            # Set up results dataset, using NDVI as a base
            who5 = np.empty(ndvi.shape, dtype=np.float32)
            who5[:] = NODATA_VAL

            valid_mask = np.empty(ndvi.shape, dtype=np.bool8)
            if ndvi is not None:
                valid_mask = ~np.isclose(ndvi, ndvi_nodata)
            ndvi_valid = ndvi[valid_mask]

            # WHO5 Calculation
            who5[valid_mask] = mean_who5 + (
                (ndvi_valid - baseline_ndvi) / ndvi_increment
            )

            return who5

        pygeoprocessing.raster_calculator(
            [
                (resample_ndvi, 1),
                (args["mean_who5"], "raw"),
                (args["baseline_ndvi"], "raw"),
                (args["ndvi_increment"], "raw"),
            ],
            who5_op,
            who5_raster,
            gdal.GDT_Float32,
            NODATA_VAL,
        )

        who5_output_rasters.append(who5_raster)

    # Total expenditures before and after
    who5_baseline_raster = who5_output_rasters.pop(0)
    baseline_label = scenario_labels.pop(0)
    for who5_raster, label in zip(who5_output_rasters, scenario_labels):
        logger.info(
            f"Calculating change in mental health expenditures for {label} scenario"
        )
        scenario_nodata = pygeoprocessing.get_raster_info(who5_raster)["nodata"][0]
        who_nodata = pygeoprocessing.get_raster_info(who5_baseline_raster)["nodata"][0]
        pop_nodata = pygeoprocessing.get_raster_info(population_resample_tif)["nodata"][
            0
        ]

        def expenditures_op(
            who5_scenario_raster,
            who5_baseline_raster,
            population_raster,
            per_capita_expenditures,
        ):
            # Set up results dataset, using NLCD as a base
            expenditures = np.empty(who5_scenario_raster.shape, dtype=np.float32)
            expenditures[:] = NODATA_VAL

            valid_mask = np.empty(who5_scenario_raster.shape, dtype=np.bool8)
            if (
                who5_scenario_raster is not None
                and who5_baseline_raster is not None
                and population_raster is not None
            ):
                valid_mask = np.logical_and(
                    ~np.isclose(who5_scenario_raster, scenario_nodata),
                    ~np.isclose(who5_baseline_raster, who_nodata),
                    ~np.isclose(population_raster, pop_nodata),
                )

            who5_scenario_valid = who5_scenario_raster[valid_mask]
            who5_baseline_valid = who5_baseline_raster[valid_mask]
            population_valid = population_raster[valid_mask]

            who5_baseline_raster[np.where((who5_baseline_raster == 0))] = 0.1
            who5_scenario_raster[np.where((who5_scenario_raster == 0))] = 0.1

            expenditures[valid_mask] = (
                (
                    (-1)
                    * (
                        (who5_scenario_valid - who5_baseline_valid)
                        / who5_baseline_valid
                    )
                )
                * population_valid
                * per_capita_expenditures
            )

            return expenditures

        expenditures_raster = os.path.join(
            args["workspace_dir"], f"expenditure_change_{label}.tif"
        )
        pygeoprocessing.raster_calculator(
            [
                (who5_raster, 1),
                (who5_baseline_raster, 1),
                (population_resample_tif, 1),
                (args["per_capita_expenditures"], "raw"),
            ],
            expenditures_op,
            expenditures_raster,
            gdal.GDT_Float32,
            NODATA_VAL,
        )

    # TODO Net present value


if __name__ == "__main__":
    # Testing
    args = {
        "workspace_dir": r"C:\Users\umn-cnootenb\Documents\NatCap\Projects\San Antonio\mental_health",
        "population_raster_path": r"G:\My Drive\Livable Cities\San Antonio\Data\population_2010_final.tif",
        "ndvi_baseline_raster_path": r"G:\My Drive\Livable Cities\San Antonio\Data\ndvi_landsat8.tif",
        "ndvi_scenario_raster_paths": [
            r"G:\My Drive\Livable Cities\San Antonio\Data\ndvi_landsat8_green.tif"
        ],
        "scenario_suffixes": ["greenspace"],
        "impact_distance_m": 1000,
        "mean_who5": 12.081,
        "baseline_ndvi": 0.097,
        "ndvi_increment": 0.136,
        "per_capita_expenditures": 356.74,
    }

    execute(args)
