from pathlib import Path
import logging
import tempfile
from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd

from osgeo import gdal
import pygeoprocessing

import natcap.invest.utils

from .. import nature
from .. import zonal_statistics

pathlike = str | Path

# Global variables
TARGET_NODATA = -1


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

BUILDING_FILENAME = "building_energy_expenditures"

CDD_FIELD = "cdd"
HDD_FIELD = "hdd"
CDD_KWH_FIELD = "cdd_kwh"
HDD_KWH_FIELD = "hdd_kwh"
CDD_COST_FIELD = "cdd_cost"
HDD_COST_FIELD = "hdd_cost"

BUILDING_TYPE_FIELD = "code"
KWH_PER_CDD_FIELD = "kwh_per_cdd"
KWH_PER_HDD_FIELD = "kwh_per_hdd"
COST_FIELD = "cost"

_EXPECTED_ENERGY_HEADERS = [
    BUILDING_TYPE_FIELD,
    KWH_PER_CDD_FIELD,
    KWH_PER_HDD_FIELD,
    COST_FIELD,
]

MORTALITY_FILENAME = "mortality_risk"

CITY_NAME_FIELD = "city"

_EXPECTED_MORTALITY_HEADERS = [
    "city",
    "t_01",
    "t_10",
    "t_mmtp",
    "t_90",
    "t_99",
    "rr_01",
    "rr_10",
    "rr_mmtp",
    "rr_90",
    "rr_99",
]


def execute(args):
    """Urban Cooling Model valuation

    Args:
        args['workspace_dir'] (string): a path to the output workspace folder.
        args['results_suffix'] (string): string appended to each output file path.
        args['city'] (string): selected city.
        args['lulc_raster_path'] (string): file path to a landcover raster.
        args['air_temp_tif'] (string): file path to an air temperature raster output from InVEST Urban Cooling Model.
        args['building_vector_path'] (string): file path to a building vector dataset. Must include field(s):
            * 'code': column linking this table to the building type associated with each polygon in the building_vector_path
        args['building_energy_table_path'] (string): file path to a table indicating the relationship between building types and
            energy use. Table headers must include:
                * 'code': column linking this table to the building type associated with each polygon in the building_vector_path
                * 'kwh_per_cdd': the energy impact of each Cooling Degree Day for this building type
                * 'kwh_per_hdd': the energy impact of each Heating Degree Day for this building type
                * 'cost_per_hdd': the cost of energy associated with this building type
        args['mortality_risk_path'] (string): file path to a table indicating the relationship between temperature and
            mortality risk for numerous cities . Table headers must include:
                * 'city': city name in the format 'City, Country'
                * 't_01': city's 1% temperature threshold
                * 't_10': city's 10% temperature threshold
                * 't_mmtp': city's minimum-mortality temperature
                * 't_90': city's 90% temperature threshold
                * 't_99': city's 99% temperature threshold
                * 'rr_01': relative mortality risk associated with t_01
                * 'rr_10': relative mortality risk associated with t_10
                * 'rr_mmtp': relative mortality risk associated with t_mmtp (i.e. 0)
                * 'rr_90': relative mortality risk associated with t_90
                * 'rr_99': relative mortality risk associated with t_99
            This comes from Guo et al. 2014.

    Returns:
        None
    """

    file_suffix = natcap.invest.utils.make_suffix_string(args, "results_suffix")

    workspace_path = Path(args["workspace_dir"])

    building_energy_df = pd.read_csv(args["building_energy_table_path"])

    # Test for correct csv headers
    for header in _EXPECTED_ENERGY_HEADERS:
        if header not in building_energy_df.columns:
            raise ValueError(
                f"Expected a header in biophysical table that matched the pattern '{header}' but was unable to find "
                f"one. Here are all the headers from {args['building_energy_table_path']}: {list(building_energy_df.columns)}"
            )

    # TODO Consider calculating HDD/CDD using WBGT, not just raw temp, to account for the lived experience of heat

    temp_dir = workspace_path / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Calculate Heating Degree Days raster
    logger.debug(f"Calculating Heating Degree Days")
    hdd_tif = workspace_path / f"hdd{file_suffix}.tif"
    hdd_calculation(args["air_temp_tif"], hdd_tif)

    # Calculate Cooling Degree Days raster
    logger.debug(f"Calculating Cooling Degree Days")
    cdd_tif = workspace_path / f"cdd{file_suffix}.tif"
    cdd_calculation(args["air_temp_tif"], cdd_tif)

    # Zonal statistics to connect Degree Days to Buildings
    building_gdf = zonal_statistics.batch_zonal_stats(
        [hdd_tif, cdd_tif],
        args["building_vector_path"],
        temp_dir,
        zonal_join_columns="mean",
        zonal_raster_labels=[f"{HDD_FIELD}{file_suffix}", f"{CDD_FIELD}{file_suffix}"],
        join_to_vector=True,
    )

    # TODO Calculate Energy Expenditures per building (REMOVE THE PICKLE STUFF bc we need to join it all to one df)
    logger.debug(f"Calculating Energy Expenditures")
    building_gdf = building_gdf.merge(
        building_energy_df, on=BUILDING_TYPE_FIELD, how="left"
    )

    # Calculating kWh per HDD/CDD
    building_gdf[HDD_KWH_FIELD + file_suffix] = (
        building_gdf[f"{HDD_FIELD}{file_suffix}__mean"]
        * building_gdf[KWH_PER_HDD_FIELD]
    )
    building_gdf[CDD_KWH_FIELD + file_suffix] = (
        building_gdf[f"{CDD_FIELD}{file_suffix}__mean"]
        * building_gdf[KWH_PER_CDD_FIELD]
    )

    # Calculating cost per HDD/CDD
    building_gdf[HDD_COST_FIELD + file_suffix] = (
        building_gdf[HDD_KWH_FIELD + file_suffix] * building_gdf[COST_FIELD]
    )
    building_gdf[CDD_COST_FIELD + file_suffix] = (
        building_gdf[CDD_KWH_FIELD + file_suffix] * building_gdf[COST_FIELD]
    )

    # Export building data
    building_gpkg = workspace_path / f"{BUILDING_FILENAME}{file_suffix}.gpkg"
    building_gdf.to_file(building_gpkg, driver="GPKG")

    # Calculate Mortality Risk
    logger.debug(f"Calculating Relative Mortality Risk")

    mortality_risk_df = pd.read_csv(
        args["mortality_risk_path"], encoding="unicode_escape"
    )

    # Test for correct csv headers
    for header in _EXPECTED_MORTALITY_HEADERS:
        if header not in mortality_risk_df.columns:
            raise ValueError(
                f"Expected a header in biophysical table that matched the pattern '{header}' but was unable to find "
                f"one. Here are all the headers from {args['mortality_risk_path']}: {list(mortality_risk_df.columns)}"
            )

    # Test if selected city is in the Guo et al dataset
    city_name_global = f"{args['city'].split(',')[0]}, USA"
    try:
        if city_name_global not in mortality_risk_df[CITY_NAME_FIELD]:
            raise IndexError("GuoStudy")
    except IndexError:
        logger.error(
            f"'{city_name_global}' not in Guo et al. 2014 Mortality Risk study"
        )
    else:
        city_mortality_risk_df = mortality_risk_df.loc[
            mortality_risk_df["city"] == city_name_global
        ]

        mortality_tif = workspace_path / f"{MORTALITY_FILENAME}{file_suffix}.tif"
        mortality_risk_calculation(
            args["air_temp_tif"], mortality_tif, city_mortality_risk_df
        )


def hdd_calculation(
    t_air_raster_path: pathlike, target_hdd_path: pathlike, hdd_threshold: float = 15.5
) -> None:
    """Raster calculator op to calculate Heating Degree Days from air temperature and HDD threshold temperature.

    Args:
        t_air_raster_path (pathlike): Pathlib path to T air raster.
        target_hdd_path (pathlike): Pathlib path to target hdd raster.
        hdd_threshold (float): number between 0-100.

    Returns:
        if T_(air,i) > hdd_threshold:
            hdd_i = 1.43 * 10**25 * T_(air,i) **-12.85
        else:
            hdd_i = -29.54 * T_(air,i) + 1941
        HOWEVER the problem is these are Fahrenheit Degree Day calculations, while the incoming temperature data is in
        Celsius. For now, we simply convert the incoming temperature data to Fahrenheit then convert the output
        Fahrenheit Degree Day to Celsius.

    """
    # Ensure path variables are Path objects
    t_air_raster_path = Path(t_air_raster_path)
    target_hdd_path = Path(target_hdd_path)

    t_air_nodata = pygeoprocessing.get_raster_info(str(t_air_raster_path))["nodata"][0]

    def hdd_op(t_air_array, hdd_val):
        hdd = np.empty(t_air_array.shape, dtype=np.float32)
        hdd[:] = TARGET_NODATA

        valid_mask = slice(None)
        if t_air_nodata is not None:
            valid_mask = ~np.isclose(t_air_array, t_air_nodata)
        t_air_valid = t_air_array[valid_mask]

        hdd[valid_mask] = np.where(
            t_air_valid > hdd_val,
            (1.43 * (10**25) * ((t_air_valid * 9 / 5 + 32) ** (-12.85))) * 5 / 9,
            ((-29.54) * (t_air_valid * 9 / 5 + 32) + 1941) * 5 / 9,
        )
        return hdd

    pygeoprocessing.raster_calculator(
        [(str(t_air_raster_path), 1), (hdd_threshold, "raw")],
        hdd_op,
        str(target_hdd_path),
        gdal.GDT_Float32,
        TARGET_NODATA,
    )


def cdd_calculation(
    t_air_raster_path: pathlike, target_cdd_path: pathlike, cdd_threshold: float = 21.1
) -> None:
    """Raster calculator op to calculate Cooling Degree Days. Currently based solely on Temperature

    Args:
        t_air_raster_path (Path): Pathlib path to T air raster.
        target_cdd_path (Path): Pathlib path to target cdd raster.
        cdd_threshold (float): number between 0-100.

    Returns:
        if T_(air,i) >= cdd_threshold:
            cdd_i = 29.58 * T_(air,i) - 1905
        else:
            cdd_i = 1.07 * 10**-18 * T_(air,i)**10.96
        HOWEVER the problem is these are Fahrenheit Degree Day calculations, while the incoming temperature data is in
        Celsius. For now, we simply convert the incoming temperature data to Fahrenheit then convert the output
        Fahrenheit Degree Day to Celsius.

    """
    t_air_nodata = pygeoprocessing.get_raster_info(str(t_air_raster_path))["nodata"][0]

    def cdd_op(t_air_array, cdd_val):
        cdd = np.empty(t_air_array.shape, dtype=np.float32)
        cdd[:] = TARGET_NODATA

        valid_mask = slice(None)
        if t_air_nodata is not None:
            valid_mask = ~np.isclose(t_air_array, t_air_nodata)
        t_air_valid = t_air_array[valid_mask]

        cdd[valid_mask] = np.where(
            t_air_valid >= cdd_val,
            (29.58 * (t_air_valid * 9 / 5 + 32) - 1905) * 5 / 9,
            (1.07 * (10 ** (-18)) * ((t_air_valid * 9 / 5 + 32) ** 10.96)) * 5 / 9,
        )
        return cdd

    pygeoprocessing.raster_calculator(
        [(str(t_air_raster_path), 1), (cdd_threshold, "raw")],
        cdd_op,
        str(target_cdd_path),
        gdal.GDT_Float32,
        TARGET_NODATA,
    )


def mortality_risk_calculation(
    t_air_raster_path: pathlike,
    target_mortality_path: pathlike,
    mortality_risk_df: pd.DataFrame,
) -> None:
    """Raster calculator op to calculate Relative Mortality Risk based on the following function:
    mortality = (Ti - T0) / (T0 - T1) * (R0 - R1) + R1
        where:
            Ti = T air raster
            T0 = upper temperature threshold
            T1 = lower temperature threshold
            R0 = upper mortality risk
            R1 = lower mortality risk

    Args:
        t_air_raster_path (pathlike): Path to T air raster.
        target_mortality_path (pathlike): Path to target mortality risk raster.
        mortality_risk_df (DataFrame): Pandas DataFrame with columns for the temperature thresholds and associated
            mortality risk

    Returns:
        None

    """
    # Ensure path variables are Path objects
    t_air_raster_path = Path(t_air_raster_path)
    target_mortality_path = Path(target_mortality_path)

    t_air_nodata = pygeoprocessing.get_raster_info(str(t_air_raster_path))["nodata"][0]

    def mortality_op(t_air_array):
        """
        Raster calculator op that calculates relative mortality risk based on temperature thresholds.
        mortality = (Ti - T0) / (T0 - T1) * (R0 - R1) + R1
        """
        mortality = np.empty(t_air_array.shape, dtype=np.float32)
        mortality[:] = TARGET_NODATA

        if t_air_nodata is not None:
            valid_mask = ~np.isclose(t_air_array, t_air_nodata)
        else:
            valid_mask = np.ones(t_air_array.shape, dtype=bool)

        thresholds = ["01", "10", "mmtp", "90", "99"]
        # Iterate through thresholds, except 99th percentile (since we linearly interpolate anything greater than 90th)
        for i, t in enumerate(thresholds[:-1]):
            i += 1

            # Temperature Thresholds
            lower_threshold = mortality_risk_df.loc[0, f"t_{t}"]
            upper_threshold = mortality_risk_df.loc[0, f"t_{thresholds[i]}"]

            # Mortality Risks
            lower_risk = mortality_risk_df.loc[0, f"rr_{t}"]
            upper_risk = mortality_risk_df.loc[0, f"rr_{thresholds[i]}"]

            # Calculate mask
            # Initial bin
            if i == 1:
                current_mask = np.logical_and(
                    valid_mask, (t_air_array < upper_threshold)
                )
            # Final Bin
            elif i == len(thresholds) - 1:
                current_mask = np.logical_and(
                    valid_mask, (t_air_array >= lower_threshold)
                )
            # All other bins
            else:
                current_mask = np.all(
                    (
                        valid_mask,
                        (t_air_array >= lower_threshold),
                        (t_air_array < upper_threshold),
                    ),
                    axis=0,
                )

            # Actual calculation
            t_air_masked = t_air_array[current_mask]
            mortality[current_mask] = (t_air_masked - upper_threshold) / (
                lower_threshold - upper_threshold
            ) * (lower_risk - upper_risk) + upper_risk

        return mortality

    pygeoprocessing.raster_calculator(
        [(str(t_air_raster_path), 1)],
        mortality_op,
        str(target_mortality_path),
        gdal.GDT_Float32,
        TARGET_NODATA,
    )
