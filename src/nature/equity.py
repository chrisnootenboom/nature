import logging
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from dotenv import load_dotenv
import os
import warnings
import itertools

import pandas as pd

import requests
import census
import us

import rasterio as rio
from osgeo import gdal, osr
import geopandas as gpd

import pygeoprocessing
import pygeoprocessing.kernels

from . import nature

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

pathlike = str | Path

# Load environment variables
load_dotenv()
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")

CENSUS_API_GEOFORMAT = {
    "state": "{:<02}",
    "county": "{:<03}",
    "tract": "{:<06}",
    "block group": "{:<01}",
}

# Create local variables
STATE_LIST = us.STATES + [us.states.DC]


def census_2020_redistricting_extraction(
    variables: List[str],
    census_block_dir: pathlike,
    workspace_path: pathlike,
    target_crs: rio.crs.CRS,
    census_api_key: str = CENSUS_API_KEY,
    state_fips_list: List[int] = None,
) -> None:
    """Extract census data from census api for 2020 redistricting by state, creating GeoPackages and CSV files.

    Args:
        variables (List[str]): List of census variables to extract.
        census_block_dir (pathlike): Path to census block shapefile directory.
        workspace_path (pathlike): Path to workspace directory.
        target_crs (rio.crs.CRS): Target crs for extracted data.
        census_api_key (str): Census API key. Defaults to CENSUS_API_KEY from .env file.
        state_fips_list (List[int], optional): List of state fips codes to extract. Defaults to None.

    Returns:
        None
    """
    # Ensure workspace path is a Path object
    workspace_path = Path(workspace_path)
    census_block_dir = Path(census_block_dir)

    # Set up census api extractor
    if state_fips_list is None:
        # Include DC in the list of states
        state_fips_list = [
            f"{state.fips:02d}" if isinstance(state.fips, int) else state.fips
            for state in STATE_LIST
        ]
    else:
        state_fips_list = [f"{state_fips:02d}" for state_fips in state_fips_list]
    for state_fips in state_fips_list:
        # Include DC in the list of states
        state = us.states.lookup(state_fips) if state_fips != "11" else us.states.DC
        logger.info(f"Extracting census data from {state.name}")
        url = f"https://api.census.gov/data/2020/dec/pl?get={','.join(variables)}&for=block:*&in=state:{state_fips}&in=county:*&in=tract:*&key={census_api_key}"

        response = requests.get(url)
        response_df = pd.DataFrame.from_dict(response.json()[1:])
        response_df.columns = response.json()[0]

        # Merge results with corresponding census block shapefile
        census_block_gdf = gpd.read_file(
            census_block_dir
            / f"tl_rd22_{state_fips}_tabblock20/tl_rd22_{state_fips}_tabblock20.shp"
        )
        census_block_gdf_merged = pd.merge(
            census_block_gdf,
            response_df,
            left_on=["STATEFP20", "COUNTYFP20", "TRACTCE20", "BLOCKCE20"],
            right_on=["state", "county", "tract", "block"],
        )

        for var in variables:
            if var == "GEO_ID":
                continue
            response_df[var] = pd.to_numeric(response_df[var])
            census_block_gdf_merged[var] = pd.to_numeric(census_block_gdf_merged[var])

        census_block_gdf_merged = census_block_gdf_merged.to_crs(target_crs)

        census_gpkg = workspace_path / f"{state.abbr}.gpkg"
        response_df.to_csv(workspace_path / f"{state.abbr}.csv")
        census_block_gdf_merged.to_file(census_gpkg, driver="GPKG")


def census_2020_population_density(
    census_block_dir: pathlike,
    workspace_path: pathlike,
    template_raster_path: pathlike,
    state_fips_list: List[int] = None,
    per_pixel: bool = False,
    mosaic: bool = True,
):
    """Calculate population density from 2020 census block data.

    Args:
        census_block_dir (pathlike): Path to census block shapefile directory.
        workspace_path (pathlike): Path to workspace directory.
        template_raster_path (pathlike): Path to template raster.
        state_fips_list (List[int], optional): List of state fips codes to extract. Defaults to None.
        per_pixel (bool, optional): Whether to calculate population per pixel. Defaults to False.
        mosaic (bool, optional): Whether to mosaic results. Defaults to True.

    Returns:
        None
    """
    # Ensure workspace path is a Path object
    workspace_path = Path(workspace_path)
    census_block_dir = Path(census_block_dir)
    template_raster_path = Path(template_raster_path)

    # Define fields
    original_population_field = "POP20"
    land_area_field = "ALAND20"
    water_area_field = "AWATER20"

    # Create temporary dir for intermediate files
    temp_dir = workspace_path / "_temp_population_density"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Get raster data
    with rio.open(template_raster_path) as src:
        target_crs = src.crs

    raster_info = pygeoprocessing.get_raster_info(str(template_raster_path))
    raster_list = []

    pixel_size, pixel_area = nature.get_mean_pixel_size_and_area(template_raster_path)

    # Create population density column (persons per sqkm)
    population_density_column = "people_per_sqkm"
    population_column = f"people_per_{int(pixel_size)}m_pixel"

    # Set up census api extractor
    if state_fips_list is None:
        # Include DC in the list of states
        state_fips_list = [
            f"{state.fips:02d}" if isinstance(state.fips, int) else state.fips
            for state in STATE_LIST
        ]
    else:
        state_fips_list = [f"{state_fips:02d}" for state_fips in state_fips_list]
    for state_fips in state_fips_list:
        # Include DC in the list of states
        state = us.states.lookup(state_fips) if state_fips != "11" else us.states.DC

        # Merge results with corresponding census block shapefile
        logger.debug(f"Extracting population data for {state.name}")
        census_block_gdf = gpd.read_file(
            census_block_dir
            / f"tl_rd22_{state_fips}_tabblock20/tl_rd22_{state_fips}_tabblock20.shp"
        )

        # Calculate population density
        census_block_gdf[population_density_column] = (
            census_block_gdf[original_population_field]
            / (census_block_gdf[land_area_field] + census_block_gdf[water_area_field])
            * 10**6
        )

        # TODO zonal stats the number of raster pixels per census geography for a more accurate measure of population per pixel

        # Calculate population per pixel
        census_block_gdf[population_column] = census_block_gdf[
            original_population_field
        ] / (
            (census_block_gdf[land_area_field] + census_block_gdf[water_area_field])
            / pixel_area
        )  # population / # of pixels in census geography

        # Project and export to GeoPackage
        census_block_gdf = census_block_gdf.to_crs(target_crs)

        census_gpkg = temp_dir / f"population_density_{state.abbr}.gpkg"
        census_block_gdf.to_file(census_gpkg, driver="GPKG")

        # Rasterize population
        logger.debug(f"Rasterizing population density for {state.name}")
        population_density_raster = temp_dir / f"{state.abbr}_density.tif"

        pygeoprocessing.create_raster_from_vector_extents(
            str(census_gpkg),
            str(population_density_raster),
            raster_info["pixel_size"],
            gdal.GDT_Int32,
            0,
        )

        pygeoprocessing.rasterize(
            str(census_gpkg),
            str(population_density_raster),
            option_list=[f"ATTRIBUTE={population_density_column}"],
        )

        # Rasterize population per pixel, if requested
        if per_pixel:
            logger.debug(f"Rasterizing population per pixel for {state.name}")
            population_raster = temp_dir / f"{state.abbr}_per_pixel.tif"

            pygeoprocessing.create_raster_from_vector_extents(
                str(census_gpkg),
                str(population_raster),
                raster_info["pixel_size"],
                gdal.GDT_Int32,
                0,
            )

            pygeoprocessing.rasterize(
                str(census_gpkg),
                str(population_raster),
                option_list=[f"ATTRIBUTE={population_column}"],
            )

        # Append to list if not HI or AK
        if state_fips not in ["15", "02"] and mosaic:
            raster_list.append(population_raster)

    if mosaic:
        # Bring results together. Outside of loop to avoid memory issues
        logger.info(f"Combining density data for all states")
        national_population_density_raster = (
            workspace_path / "population_density_2020.tif"
        )
        nature.mosaic_raster(raster_list, national_population_density_raster)

        if per_pixel:
            logger.info(f"Combining population data for all states")
            national_population_raster = workspace_path / "population_2020.tif"
            nature.mosaic_raster(raster_list, national_population_raster)


def acs5_api_extraction(
    aoi_vector_path: pathlike,
    acs_codes: List[str] | pd.Series,
    year: int,
    census_geo_type: str,
    census_file_format: Dict,
    workspace_path: pathlike,
    census_api_key: str = CENSUS_API_KEY,
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Extract ACS 5-year summary tables from the Census API.

    Args:
        aoi_vector_path (pathlike): Path to AOI shapefile.
        acs_codes (List[str] or pd.Series): List or Series of ACS codes to extract.
        year (int): Year of ACS data to extract.
        census_geo_type (str): Census geography type to extract.
        census_file_format (Dict): Dictionary of census file information.
        census_geoformat (Dict): Dictionary of census geographic format information.
        workspace_path (pathlike): Path to workspace directory.
        census_api_key (str): Census API key. Defaults to CENSUS_API_KEY from .env file.

    Returns:
        census_df (pd.DataFrame): DataFrame of extracted census data.
        census_gdf (gpd.GeoDataFrame): GeoDataFrame of extracted census data.
    """

    # Ensure workspace path is a Path object
    workspace_path = Path(workspace_path)
    aoi_vector_path = Path(aoi_vector_path)

    # Ensure list of acs codes
    acs_codes = list(acs_codes)

    # Load census and AOI geographies
    census_gdf = gpd.read_file(census_file_format["file"])
    aoi_gdf = gpd.read_file(aoi_vector_path)

    # Get original CRS data on AOI and census geographies
    aoi_crs = aoi_gdf.crs
    census_crs = census_gdf.crs

    # Project AOI to match census geographies
    aoi_projected_gdf = aoi_gdf.to_crs(census_crs)

    temp_dir = workspace_path / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    aoi_projected_path = temp_dir / "aoi_inequality_projected.gpkg"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aoi_projected_gdf.to_file(aoi_projected_path, driver="GPKG")

    # Get FIPS for geographic extraction (i.e. we need to know which counties to extract for the city)
    geo_fields = census_file_format.copy()
    geo_fields.pop("file")
    geo_ids = nature.extract_admin_intersections(
        aoi_projected_path, census_file_format["file"], geo_fields
    )
    # Ensure proper string formatting for API
    geo_ids = {
        scale: [CENSUS_API_GEOFORMAT[scale].format(item) for item in geo_ids[scale]]
        for scale in list(geo_ids)
    }

    # Create archive of spatial units (for filtering later)
    spatial_units = pd.DataFrame(geo_ids)

    # Set up census api extractor
    c = census.Census(census_api_key)
    results = []

    # Iterate through spatial units
    census_geographies = ["state", "county", "tract", "block group", "block"]
    relevant_census_geographies = census_geographies[
        : census_geographies.index(census_geo_type) + 1
    ]

    extraction_geometry = (
        relevant_census_geographies
        if census_geo_type == "county"
        else relevant_census_geographies[:-1]
    )
    census_spatial_units = spatial_units[extraction_geometry]
    census_spatial_units = census_spatial_units.drop_duplicates()
    census_spatial_units = census_spatial_units.reset_index(drop=True)

    for ind, row in census_spatial_units.iterrows():
        logger.info(
            f"Extracting census data from {extraction_geometry[-1]}: "
            f"{int(ind) + 1} of {len(census_spatial_units)}"
        )

        # Extract census data
        # ACS 5-year summary tables
        if census_geo_type == "county":
            response = c.acs5.state_county(
                acs_codes,
                year=year,
                state_fips=row["state"],
                county_fips=row["county"],
            )
        elif census_geo_type == "tract":
            response = c.acs5.state_county_tract(
                acs_codes,
                year=year,
                state_fips=row["state"],
                county_fips=row["county"],
                tract=census.Census.ALL,
            )
        elif census_geo_type == "block group":
            response = c.acs5.state_county_blockgroup(
                acs_codes,
                year=year,
                state_fips=row["state"],
                county_fips=row["county"],
                tract=row["tract"],
                blockgroup=census.Census.ALL,
            )
        else:
            break

        # Store census data
        response_df = pd.DataFrame(response)
        results.append(response_df)

    # Convert to DataFrame and move the geographic columns to the front
    results_df = pd.concat(results).reset_index(drop=True)
    if "GEO_ID" not in results_df.columns:
        results_df["GEO_ID"] = ""
        for census_geo in relevant_census_geographies:
            results_df["GEO_ID"] += results_df[census_geo].map(str)
    geo_col = relevant_census_geographies.copy()
    geo_col.insert(0, "GEO_ID")
    geo_col.reverse()

    for col_name in geo_col:
        first_col = results_df.pop(col_name)
        results_df.insert(0, col_name, first_col)

    # Merge census data and GIS data
    logger.info(f"Merging with census {census_geo_type} GIS data")
    census_gdf = pd.merge(
        census_gdf,
        results_df,
        left_on=[geo_fields[i] for i in relevant_census_geographies],
        right_on=relevant_census_geographies,
    )
    census_gdf = census_gdf.clip(aoi_projected_gdf)

    # Project to original AOI coordinate system
    census_gdf = census_gdf.to_crs(aoi_crs)

    # Create dataframe without geographic data
    census_df = census_gdf[results_df.columns]

    return census_df, census_gdf


def extract_and_process_acs_data(
    aoi_vector_path: Path,
    acs_codes_df: pd.DataFrame,
    acs_year: int,
    census_geo_type,
    census_file_format,
    workspace_path,
    file_suffix="",
) -> Tuple[Path, Path]:
    """Extracts ACS data from the Census API, given a study area (aoi_vector_path) and selected census datatypes (acs_codes_df).
    Census geographic data is output sans variables as a single GeoPackage. Census variable data for the selected geographies are
    are grouped by variable structure (e.g., "Race" vs "Race | Income | Food Stamps") and exported to a csv file. For variable
    structures with multiple components (e.g., "Race | Income | Food Stamps"), the data are exported to a csv file for each
    combinatorial overlay (e.g., "Race", "Race | Income", "Race | Food Stamps", "Income", "Income | Food Stamps", etc).

    Args:
        aoi_vector_path (pathlib.Path): Path to AOI vector file.
        acs_codes_df (pandas.DataFrame): Pandas dataframe containing ACS codes.
        acs_year (int): Year of ACS data to extract.
        census_file_format (str): Format of Census API data to extract (i.e., file location, field names, etc).
        census_geo_type (str): Type of Census geography to extract.
        census_api_key (str): Census API key.
        workspace_path (str): Path to workspace.
        file_suffix (str, optional): Suffix to append to output file names. Defaults to "".

    Returns:
        acs_gpkg (pathlib.Path): Path to ACS GeoPackage.
        acs_dir (pathlib.Path): Path to ACS directory.
    """

    # Ensure Path variables are Path objects
    aoi_vector_path = Path(aoi_vector_path)

    # Raise error if chosen Census geo-type is not a valid option (e.g. state, county, tract, or block group)
    if census_geo_type not in nature.equity.CENSUS_API_GEOFORMAT.keys():
        raise ValueError(
            f"The chosen Census geography scale is not in ['state', 'county', 'tract', 'block group']"
        )

    # Create file suffix
    file_suffix = nature.make_scenario_suffix(file_suffix)

    # Create ACS workspace
    acs_dir = Path(workspace_path) / f"acs_{acs_year}"
    acs_dir.mkdir(parents=True, exist_ok=True)

    # Census ACS 5yr data extraction
    acs_codes_list = list(acs_codes_df["variable"])
    acs_df, acs_gdf = nature.equity.acs5_api_extraction(
        aoi_vector_path,
        acs_codes_list,
        acs_year,
        census_geo_type,
        census_file_format,
        workspace_path,
    )

    # Export census geometry and other geospatial information to shapefile
    acs_geo_gdf = acs_gdf.loc[:, ~acs_gdf.columns.isin(acs_codes_list)]
    acs_shp, acs_gpkg = nature.geospatial_export(
        acs_geo_gdf,
        f"acs_{census_geo_type.replace(' ','_')}{file_suffix}",
        workspace_path,
    )

    acs_fips_df = acs_df.drop(acs_codes_list, axis=1)
    acs_vars_df = acs_df[acs_codes_list]

    # Export extracted census data to table
    acs_raw_csv = (
        acs_dir / f"acs_{census_geo_type.replace(' ','_')}_raw_data{file_suffix}.csv"
    )
    acs_df.to_csv(acs_raw_csv, index=False)
    acs_fips_csv = (
        acs_dir / f"acs_{census_geo_type.replace(' ','_')}_FIPS{file_suffix}.csv"
    )
    acs_fips_df.to_csv(acs_fips_csv, index=False)

    acs_output_list = []

    # Iterate through variable groupings to create renaming column
    for structure in acs_codes_df["structure"].unique():
        logger.debug(f"Extracting ACS data on {structure.replace('|', ' AND ')}")
        # Subset ACS codes based on selected 'structure' (e.g. "food stamps | race")
        selected_structure_df = acs_codes_df[
            acs_codes_df["structure"] == structure
        ].copy()

        # Iterate through datatypes within structure (e.g. Count, Median Income)
        for datatype in selected_structure_df["datatype"].unique():
            selected_acs_codes_df = selected_structure_df[
                selected_structure_df["datatype"] == datatype
            ].copy()

            # Identify data overlays within the structure (e.g. "food stamps" and "race")
            overlays = structure.split("|")
            overlays.reverse()

            # Create renaming dictionary for dataframe columns based on structure and overlays
            selected_acs_codes_df["RENAME"] = ""
            for index, overlay in enumerate(overlays):
                if overlay in selected_acs_codes_df.columns:
                    # Test column for NaN
                    nan_col = selected_acs_codes_df[overlay].isnull()
                    overlay_column = selected_acs_codes_df[overlay].fillna("").map(str)
                    if index != 0:
                        overlay_column[~nan_col] = " | " + overlay_column[~nan_col]
                    selected_acs_codes_df["RENAME"] += overlay_column
                else:
                    if index != 0:
                        overlay = " | " + overlay
                    selected_acs_codes_df["RENAME"] += str(overlay)
            # Trim the start of strings to remove " | "
            selected_acs_codes_df.loc[
                selected_acs_codes_df["RENAME"].str.startswith(" | "), "RENAME"
            ] = selected_acs_codes_df.loc[
                selected_acs_codes_df["RENAME"].str.startswith(" | "), "RENAME"
            ].str[
                3:
            ]

            # Renaming dictionary
            renaming_dict = pd.Series(
                selected_acs_codes_df["RENAME"].values,
                index=selected_acs_codes_df["variable"],
            ).to_dict()

            # Subset ACS variable data by the selected ACS codes and rename the output columns
            substructure_df = acs_vars_df[list(selected_acs_codes_df["variable"])]
            substructure_df = substructure_df.rename(columns=renaming_dict)

            # Remove unlabeled columns
            substructure_df = substructure_df.loc[
                :, ~substructure_df.columns.isin([""])
            ]
            acs_output_list.append(substructure_df)

            # Join ACS spatial data (e.g. parcel ID, state FIPS, geometry) to renamed ACS variable data
            output_df = pd.concat([acs_fips_df, substructure_df], axis=1)
            output_gdf = gpd.GeoDataFrame(
                pd.concat([pd.DataFrame(acs_geo_gdf), substructure_df], axis=1)
            )

            # Create naming convention
            output_name = f"acs_{census_geo_type.replace(' ','_')}_{structure.replace('|', '_')}{file_suffix}"

            # Create folder for combinatorial overlay analysis if the datatype is COUNT and can be so recombined
            if datatype == "Count" and "|" in structure:
                # Create overlay subdirectory
                structure_dir = Path(acs_dir) / structure.replace("|", "_")
                structure_dir.mkdir(parents=True, exist_ok=True)

                # Write full overlay output file to csv
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output_csv = structure_dir / f"{output_name}.csv"
                    output_df.to_csv(output_csv, index=False)

                # Write full overlay output file to shp and gpkg
                # output_shp, output_gpkg = nature.geospatial_export(
                #     output_gdf, output_name, acs_dir
                # )

                # Reconfigure ACS data by overlay type combinatorially
                structure_list = structure.split("|")
                structure_list.reverse()

                data_cols = [col.split(" | ") for col in substructure_df]

                for i in range(len(structure_list) - 1):
                    for combination in itertools.combinations(structure_list, i + 1):
                        sub_output_name = f"acs_{census_geo_type.replace(' ','_')}_{'_'.join(combination)}{file_suffix}"
                        output_csv = structure_dir / f"{sub_output_name}.csv"
                        if output_csv.exists():
                            continue
                        logger.debug(f"Creating overlay of {' BY '.join(combination)}")
                        max_index = max([structure_list.index(i) for i in combination])
                        selected_cols = pd.DataFrame(
                            [l for l in data_cols if len(l) == (max_index + 1)],
                            columns=structure_list[: max_index + 1],
                        )
                        selected_cols["SUBSET"] = selected_cols[
                            list(combination)
                        ].apply(lambda row: " | ".join(row.values.astype(str)), axis=1)
                        selected_cols["COLUMN"] = selected_cols[
                            structure_list[: max_index + 1]
                        ].apply(lambda row: " | ".join(row.values.astype(str)), axis=1)

                        substructure_copy_df = substructure_df.copy()
                        for col in selected_cols["SUBSET"].unique():
                            substructure_copy_df[col] = substructure_copy_df[
                                list(
                                    selected_cols[selected_cols["SUBSET"] == col][
                                        "COLUMN"
                                    ]
                                )
                            ].sum(axis=1)
                        substructure_copy_df = substructure_copy_df[
                            list(selected_cols["SUBSET"].unique())
                        ]

                        # Join ACS spatial data (e.g. parcel ID, state FIPS, geometry) to renamed ACS variable data
                        output_df = pd.concat(
                            [acs_fips_df, substructure_copy_df], axis=1
                        )
                        output_gdf = gpd.GeoDataFrame(
                            pd.concat(
                                [pd.DataFrame(acs_geo_gdf), substructure_copy_df],
                                axis=1,
                            )
                        )

                        # Write data to csv
                        substructure_copy_df.to_csv(output_csv, index=False)

                        # Write data to shp and gpkg
                        # output_shp, output_gpkg = nature.geospatial_export(
                        #     output_gdf, sub_output_name, structure_dir
                        # )

            else:
                # Write output files to csv
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output_csv = acs_dir / f"{output_name}.csv"
                    output_df.to_csv(output_csv, index=False)

                # Write output files file to shp and gpkg
                # output_shp, output_gpkg = nature.geospatial_export(
                #     output_gdf, output_name, acs_dir
                # )

    return (acs_gpkg, acs_dir)
