import yaml
import os
from pathlib import Path
import datetime
import logging
import warnings
from dotenv import load_dotenv
import tempfile

import numpy as np
import pandas as pd

import geopandas as gpd
from osgeo import gdal

import degreedays
import degreedays.api.data

import pygeoprocessing
import natcap.invest.urban_cooling_model

from . import ucm_valuation
from .. import nature
from .. import zonal_statistics
from .. import rasterops
from .. import model_metadata

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()
DEGREEDAY_API_ACCOUNT_KEY = os.getenv("DEGREEDAY_API_ACCOUNT_KEY")
DEGREEDAY_API_SECURITY_KEY = os.getenv("DEGREEDAY_API_SECURITY_KEY")

# Global variables
TARGET_NODATA = -1

ANNUAL_SUFFIX = "annual"


def degree_days_api(
    centroid_gdf: gpd.GeoDataFrame,
    year: int,
    hdd_threshold=15.5,
    cdd_threshold=21.1,
    degreeday_api_account_key: str = DEGREEDAY_API_ACCOUNT_KEY,
    degreeday_api_security_key: str = DEGREEDAY_API_SECURITY_KEY,
):
    """Extract heating and cooling degree days from the Degree Days API.

    Parameters:
        centroid_gdf (GeoDataFrame): a GeoDataFrame containing a single point geometry
        year (int): the year in question.
        security_key (str): the security key for the Degree Days API
        hdd_threshold (float): the heating degree day threshold
        cdd_threshold (float): the cooling degree day threshold
        degreeday_api_account_key (str): the account key for the Degree Days API

    Returns:
        Two Series containing heating and cooling degree days, respectively.
    """

    api = degreedays.api.DegreeDaysApi.fromKeys(
        degreedays.api.AccountKey(degreeday_api_account_key),
        degreedays.api.SecurityKey(degreeday_api_security_key),
    )

    time_period = degreedays.api.data.Period.dayRange(
        degreedays.time.DayRange(datetime.date(year, 1, 1), datetime.date(year, 12, 31))
    )

    hdd_spec = degreedays.api.data.DataSpec.dated(
        degreedays.api.data.Calculation.heatingDegreeDays(
            degreedays.api.data.Temperature.celsius(hdd_threshold)
        ),
        degreedays.api.data.DatedBreakdown.monthly(time_period),
    )

    cdd_spec = degreedays.api.data.DataSpec.dated(
        degreedays.api.data.Calculation.coolingDegreeDays(
            degreedays.api.data.Temperature.celsius(cdd_threshold)
        ),
        degreedays.api.data.DatedBreakdown.monthly(time_period),
    )

    request = degreedays.api.data.LocationDataRequest(
        degreedays.api.data.Location.longLat(
            degreedays.geo.LongLat(centroid_gdf.x, centroid_gdf.y)
        ),
        degreedays.api.data.DataSpecs(hdd_spec, cdd_spec),
    )

    response = api.dataApi.getLocationData(request)

    hdd_data = response.dataSets[hdd_spec]
    cdd_data = response.dataSets[cdd_spec]

    return hdd_data, cdd_data


def execute(args):
    """Execute the Monthly Urban Cooling Model and Valuation.

    Args:

        args['workspace_dir'] (string): a path to the output workspace folder.
        args['results_suffix'] (string): string appended to each output file path.

        args['n_workers']
        args['lulc_raster_path']
        args['ref_eto_raster_path']
        args['aoi_vector_path']
        args['biophysical_table_path']
        args['green_area_cooling_distance']
        args['t_air_average_radius']
        args['uhi_max']  # TODO: Make this seasonal?
        args['building_vector_path']
        args['energy_consumption_table_path']
        args['cc_method']
        args['cc_weight_shade']
        args['cc_weight_albedo']
        args['cc_weight_eti']

        args['air_temp_dir'] (string): Path to a folder containing temperature data from SOURCE.
        args['year'] (int): year of interest (between 1900 and 2017).

        args['city'] (string): selected city.
        args['building_energy_table_path'] (string): file path to a table indicating the relationship between LULC classes and
            energy use. Table headers must include:
                * 'lucode': column linking this table to the lulc_raster_path raster classes
                * 'building type': the building type associated with each raster lucode
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

    workspace_path = Path(args["workspace_dir"])

    building_energy_df = pd.read_csv(args["building_energy_table_path"])

    mortality_risk_df = pd.read_csv(
        args["mortality_risk_path"], encoding="unicode_escape"
    )

    # Test for correct csv headers in energy file
    for header in ucm_valuation._EXPECTED_ENERGY_HEADERS:
        if header not in building_energy_df.columns:
            raise ValueError(
                f"Expected a header in biophysical table that matched the pattern '{header}' but was unable to find "
                f"one. Here are all the headers from {args['building_energy_table_path']}: {list(building_energy_df.columns)}"
            )

    for header in ucm_valuation._EXPECTED_MORTALITY_HEADERS:
        if header not in mortality_risk_df.columns:
            raise ValueError(
                f"Expected a header in biophysical table that matched the pattern '{header}' but was unable to find "
                f"one. Here are all the headers from {args['mortality_risk_path']}: {list(mortality_risk_df.columns)}"
            )

    # Test if selected city is in the Guo et al dataset
    city_name_global = f"{args['city'].split(',')[0]}, USA"
    try:
        if city_name_global not in mortality_risk_df[ucm_valuation.CITY_NAME_FIELD]:
            raise IndexError("GuoStudy")
    except IndexError:
        logger.error(
            f"'{city_name_global}' not in Guo et al. 2014 Mortality Risk study"
        )

    # Extract expected baseline temperature (degrees C) for the city
    monthly_temperatures = nature.extract_monthly_temperatures(
        Path(args["air_temp_dir"]), args["aoi_vector_path"], year=args["year"]
    )

    ucm_args = {
        k: args[k]
        for k in args.keys()
        if k in model_metadata.NCI_MODELS["urban_cooling_model"].args
    }

    ucm_args["do_energy_valuation"] = False
    ucm_args["do_productivity_valuation"] = False
    ucm_args["avg_rel_humidity"] = 0.0

    valuation_args = {
        k: args[k]
        for k in args.keys()
        if k in model_metadata.NCI_MODELS["urban_cooling_valuation"].args
    }

    # Monthly UCM
    month_suffixes = []
    result_building_gdf = None
    for month, row in monthly_temperatures.iterrows():
        logger.info(f"{month} ({row['temperature']} C)")

        month_suffix = nature.make_scenario_suffix(args["results_suffix"], month)
        month_suffixes.append(month_suffix)

        ucm_args["t_ref"] = row["temperature"]
        ucm_args["results_suffix"] = month_suffix
        natcap.invest.urban_cooling_model.execute(ucm_args)

        # Valuation
        valuation_args["results_suffix"] = month_suffix
        valuation_args["air_temp_tif"] = (
            Path(args["workspace_dir"]) / f"intermediate/T_air{month_suffix}.tif"
        )

        # Script from valuation model, here to avoid multiple building results dataframes

        # TODO Consider calculating HDD/CDD using WBGT, not just raw temp, to account for the lived experience of heat

        temp_dir = workspace_path / "temp"
        temp_dir.mkdir(exist_ok=True)

        # Calculate Heating Degree Days raster
        logger.debug(f"Calculating Heating Degree Days")
        hdd_tif = workspace_path / f"hdd{month_suffix}.tif"
        ucm_valuation.hdd_calculation(valuation_args["air_temp_tif"], hdd_tif)

        # Calculate Cooling Degree Days raster
        logger.debug(f"Calculating Cooling Degree Days")
        cdd_tif = workspace_path / f"cdd{month_suffix}.tif"
        ucm_valuation.cdd_calculation(valuation_args["air_temp_tif"], cdd_tif)

        # Zonal statistics to connect Degree Days to Buildings
        building_gdf = zonal_statistics.batch_zonal_stats(
            [hdd_tif, cdd_tif],
            valuation_args["building_vector_path"],
            temp_dir,
            zonal_join_columns=["mean"],
            zonal_raster_labels=[
                f"{ucm_valuation.HDD_FIELD}{month_suffix}",
                f"{ucm_valuation.CDD_FIELD}{month_suffix}",
            ],
            join_to_vector=True,
        )

        # TODO Calculate Energy Expenditures per building (REMOVE THE PICKLE STUFF bc we need to join it all to one df)
        logger.debug(f"Calculating Energy Expenditures")
        building_gdf = building_gdf.merge(
            building_energy_df, on=ucm_valuation.BUILDING_TYPE_FIELD, how="left"
        )

        # Calculating kWh per HDD/CDD
        building_gdf[ucm_valuation.HDD_KWH_FIELD + month_suffix] = (
            building_gdf[f"{ucm_valuation.HDD_FIELD}{month_suffix}__mean"]
            * building_gdf[ucm_valuation.KWH_PER_HDD_FIELD]
        )
        building_gdf[ucm_valuation.CDD_KWH_FIELD + month_suffix] = (
            building_gdf[f"{ucm_valuation.CDD_FIELD}{month_suffix}__mean"]
            * building_gdf[ucm_valuation.KWH_PER_CDD_FIELD]
        )

        # Calculating cost per HDD/CDD
        building_gdf[ucm_valuation.HDD_COST_FIELD + month_suffix] = (
            building_gdf[ucm_valuation.HDD_KWH_FIELD + month_suffix]
            * building_gdf[ucm_valuation.COST_FIELD]
        )
        building_gdf[ucm_valuation.CDD_COST_FIELD + month_suffix] = (
            building_gdf[ucm_valuation.CDD_KWH_FIELD + month_suffix]
            * building_gdf[ucm_valuation.COST_FIELD]
        )

        # Export building data
        if result_building_gdf is None:
            result_building_gdf = building_gdf.copy()
        else:
            cols_to_use = building_gdf.columns.difference(result_building_gdf.columns)
            result_building_gdf = result_building_gdf.merge(
                building_gdf[cols_to_use], left_index=True, right_index=True, how="left"
            )

        # Calculate Mortality Risk
        if city_name_global in mortality_risk_df[ucm_valuation.CITY_NAME_FIELD]:
            logger.debug(f"Calculating Relative Mortality Risk")
            city_mortality_risk_df = mortality_risk_df.loc[
                mortality_risk_df["city"] == city_name_global
            ]

            mortality_tif = (
                workspace_path / f"{ucm_valuation.MORTALITY_FILENAME}{month_suffix}.tif"
            )
            ucm_valuation.mortality_risk_calculation(
                valuation_args["air_temp_tif"], mortality_tif, city_mortality_risk_df
            )

    # Annualize heating and cooling degree days
    annual_suffix = nature.make_scenario_suffix(
        valuation_args["results_suffix"], ANNUAL_SUFFIX
    )

    # Annualize heating and cooling degree days
    for field in [
        ucm_valuation.HDD_FIELD,
        ucm_valuation.CDD_FIELD,
    ]:
        result_building_gdf[f"{field}{annual_suffix}"] = result_building_gdf[
            [f"{field}{month_suffix}__mean" for month_suffix in month_suffixes]
        ].sum(axis=1)

    # Annualize kWh and cost from heating and cooling degree days
    for field in [
        ucm_valuation.HDD_KWH_FIELD,
        ucm_valuation.CDD_KWH_FIELD,
        ucm_valuation.HDD_COST_FIELD,
        ucm_valuation.CDD_COST_FIELD,
    ]:
        result_building_gdf[f"{field}{annual_suffix}"] = result_building_gdf[
            [f"{field}{month_suffix}" for month_suffix in month_suffixes]
        ].sum(axis=1)

    # Export building data
    file_suffix = nature.make_scenario_suffix(args["results_suffix"])
    results_building_gpkg = (
        workspace_path / f"{ucm_valuation.BUILDING_FILENAME}{file_suffix}.gpkg"
    )
    result_building_gdf.to_file(results_building_gpkg, driver="GPKG")
