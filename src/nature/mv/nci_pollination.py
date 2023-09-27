import hashlib
import inspect
import logging
import numpy as np
import collections
import re
import itertools
from pathlib import Path
from osgeo import gdal, ogr

from natcap.invest import utils, spec_utils, pollination
from natcap.invest.unit_registry import u
import pygeoprocessing
import taskgraph
from .. import nature
from .. import rasterops
from .model_metadata import NCI_METADATA

LOGGER = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
LOGGER.addHandler(handler)
LOGGER.setLevel(logging.DEBUG)

# TODO update this to the actual model spec
MODEL_SPEC = {
    "model_name": NCI_METADATA["pollination_mv"].model_title,
    "pyname": NCI_METADATA["pollination_mv"].pyname,
    "args": {
        "workspace_dir": spec_utils.WORKSPACE,
        "results_suffix": spec_utils.SUFFIX,
        "n_workers": spec_utils.N_WORKERS,
        "landcover_raster_path": {
            **spec_utils.LULC,
            "projected": True,
            "about": "Map of LULC codes. All values in this raster must have "
            "corresponding entries in the Biophysical Table.",
        },
        "guild_table_path": {
            "type": "csv",
            "index_col": "species",
            "columns": {
                "species": {
                    "type": "freestyle_string",
                    "about": "Unique name or identifier for each pollinator "
                    "species or guild of interest.",
                },
                "nesting_suitability_[SUBSTRATE]_index": {
                    "type": "ratio",
                    "about": "Utilization of the substrate by this species, where "
                    "1 indicates the nesting substrate is fully utilized "
                    "and 0 indicates it is not utilized at all. Replace "
                    "[SUBSTRATE] with substrate names matching those in "
                    "the Biophysical Table, so that there is a column for "
                    "each substrate.",
                },
                "foraging_activity_[SEASON]_index": {
                    "type": "ratio",
                    "about": "Pollinator activity for this species/guild in each "
                    "season. 1 indicates maximum activity for the "
                    "species/guild, and 0 indicates no activity. Replace "
                    "[SEASON] with season names matching those in the "
                    "biophysical table, so that there is a column for "
                    "each season.",
                },
                "alpha": {
                    "type": "number",
                    "units": u.meters,
                    "about": "Average distance that this species or guild travels "
                    "to forage on flowers.",
                },
                "relative_abundance": {
                    "type": "ratio",
                    "about": "The proportion of total pollinator abundance that "
                    "consists of this species/guild.",
                },
            },
            "about": "A table mapping each pollinator species or guild of interest "
            "to its pollination-related parameters.",
            "name": "Guild Table",
        },
        "landcover_biophysical_table_path": {
            "type": "csv",
            "index_col": "lucode",
            "columns": {
                # TODO Add new columns
                "lucode": spec_utils.LULC_TABLE_COLUMN,
                "nesting_[SUBSTRATE]_availability_index": {
                    "type": "ratio",
                    "about": "Index of availability of the given substrate in this "
                    "LULC class. Replace [SUBSTRATE] with substrate names "
                    "matching those in the Guild Table, so that there is "
                    "a column for each substrate.",
                },
                "floral_resources_[SEASON]_index": {
                    "type": "ratio",
                    "about": "Abundance of flowers during the given season in this "
                    "LULC class. This is the proportion of land area "
                    "covered by flowers, multiplied by the proportion of "
                    "the season for which there is that coverage. Replace "
                    "[SEASON] with season names matching those in the "
                    "Guild Table, so that there is a column for each "
                    "season.",
                },
            },
            "about": "A table mapping each LULC class to nesting availability and "
            "floral abundance data for each substrate and season in that "
            "LULC class. All values in the LULC raster must have "
            "corresponding entries in this table.",
            "name": "biophysical table",
        },
        # TODO change farm vector stuff, once we know what we want from the farm vector
        "farm_vector_path": {
            "type": "vector",
            "fields": {
                "crop_type": {
                    "type": "freestyle_string",
                    "about": "Name of the crop grown on each polygon, e.g. "
                    "'blueberries', 'almonds', etc.",
                },
                "half_sat": {
                    "type": "ratio",
                    "about": "The half saturation coefficient for the crop grown "
                    "in this area. This is the wild pollinator abundance "
                    "(i.e. the proportion of all pollinators that are "
                    "wild) needed to reach half of the total potential "
                    "pollinator-dependent yield.",
                },
                "season": {
                    "type": "freestyle_string",
                    "about": "The season in which the crop is pollinated. Season "
                    "names must match those in the Guild Table and "
                    "Biophysical Table.",
                },
                "fr_[SEASON]": {  # floral resources for each season
                    "type": "ratio",
                    "about": "The floral resources available at this farm for the "
                    "given season. Replace [SEASON] with season names "
                    "matching those in the Guild Table and Biophysical "
                    "Table, so that there is one field for each season.",
                },
                "n_[SUBSTRATE]": {  # nesting availabilities for each substrate
                    "type": "ratio",
                    "about": "The nesting suitability for the given substrate at "
                    "this farm. given substrate. Replace [SUBSTRATE] with "
                    "substrate names matching those in the Guild Table "
                    "and Biophysical Table, so that there is one field "
                    "for each substrate.",
                },
                "p_dep": {
                    "type": "ratio",
                    "about": "The proportion of crop dependent on pollinators.",
                },
                "p_managed": {
                    "type": "ratio",
                    "about": "The proportion of pollination required on the farm "
                    "that is provided by managed pollinators.",
                },
            },
            "geometries": spec_utils.POLYGONS,
            "required": False,
            "about": "Map of farm sites to be analyzed, with pollination data "
            "specific to each farm.",
            "name": "farms map",
        },
    },
    "outputs": {
        # TODO Update this to the actual outputs
        "farm_results.shp": {
            "created_if": "farm_vector_path",
            "about": "A copy of the input farm polygon vector file with additional fields",
            "geometries": spec_utils.POLYGONS,
            "fields": {
                "p_abund": {
                    "about": (
                        "Average pollinator abundance on the farm for the "
                        "active season"
                    ),
                    "type": "ratio",
                },
                "y_tot": {
                    "about": (
                        "Total yield index, including wild and managed "
                        "pollinators and pollinator independent yield."
                    ),
                    "type": "ratio",
                },
                "pdep_y_w": {
                    "about": (
                        "Proportion of potential pollination-dependent yield "
                        "attributable to wild pollinators."
                    ),
                    "type": "ratio",
                },
                "y_wild": {
                    "about": (
                        "Proportion of the total yield attributable to wild "
                        "pollinators."
                    ),
                    "type": "ratio",
                },
            },
        },
        "farm_pollinators.tif": {
            "created_if": "farm_vector_path",
            "about": "Total pollinator abundance across all species per season, "
            "clipped to the geometry of the farm vector’s polygons.",
            "bands": {1: {"type": "ratio"}},
        },
        "pollinator_abundance_[SPECIES]_[SEASON].tif": {
            "about": "Abundance of pollinator SPECIES in season SEASON.",
            "bands": {1: {"type": "ratio"}},
        },
        "pollinator_supply_[SPECIES].tif": {
            "about": "Index of pollinator SPECIES that could be on a pixel given "
            "its arbitrary abundance factor from the table, multiplied by "
            "the habitat suitability for that species at that pixel, "
            "multiplied by the available floral resources that a "
            "pollinator could fly to from that pixel.",
            "bands": {1: {"type": "ratio"}},
        },
        "total_pollinator_abundance_[SEASON].tif": {
            "created_if": "farm_vector_path",
            "about": "Total pollinator abundance across all species per season.",
            "bands": {1: {"type": "ratio"}},
        },
        "total_pollinator_yield.tif": {
            "created_if": "farm_vector_path",
            "about": "Total pollinator yield index for pixels that overlap farms, "
            "including wild and managed pollinators.",
            "bands": {1: {"type": "ratio"}},
        },
        "wild_pollinator_yield.tif": {
            "created_if": "farm_vector_path",
            "about": "Pollinator yield index for pixels that overlap farms, for "
            "wild pollinators only.",
            "bands": {1: {"type": "ratio"}},
        },
        "intermediate_outputs": {
            "type": "directory",
            "contents": {
                "blank_raster.tif": {
                    "about": (
                        "Blank raster used for rasterizing all the farm parameters/fields later"
                    ),
                    "bands": {1: {"type": "integer"}},
                },
                "convolve_ps_[SPECIES].tif": {
                    "about": "Convolved pollinator supply",
                    "bands": {1: {"type": "ratio"}},
                },
                "farm_nesting_substrate_index_[SUBSTRATE].tif": {
                    "about": "Rasterized substrate availability",
                    "bands": {1: {"type": "ratio"}},
                },
                "farm_pollinator_[SEASON].tif": {
                    "about": "On-farm pollinator abundance",
                    "bands": {1: {"type": "ratio"}},
                },
                "farm_relative_floral_abundance_index_[SEASON].tif": {
                    "about": "On-farm relative floral abundance",
                    "bands": {1: {"type": "ratio"}},
                },
                "floral_resources_[SPECIES].tif": {
                    "about": "Floral resources available to the species",
                    "bands": {1: {"type": "ratio"}},
                },
                "foraged_flowers_index_[SPECIES]_[SEASON].tif": {
                    "about": ("Foraged flowers index for the given species and season"),
                    "bands": {1: {"type": "ratio"}},
                },
                "habitat_nesting_index_[SPECIES].tif": {
                    "about": "Habitat nesting index for the given species",
                    "bands": {1: {"type": "ratio"}},
                },
                "half_saturation_[SEASON].tif": {
                    "about": "Half saturation constant for the given season",
                    "bands": {1: {"type": "ratio"}},
                },
                "kernel_[ALPHA].tif": {
                    "about": "Exponential decay kernel for the given radius",
                    "bands": {1: {"type": "ratio"}},
                },
                "local_foraging_effectiveness_[SPECIES].tif": {
                    "about": "Foraging effectiveness for the given species",
                    "bands": {1: {"type": "ratio"}},
                },
                "managed_pollinators.tif": {
                    "about": "Managed pollinators rasterized from the farm vector",
                    "bands": {1: {"type": "ratio"}},
                },
                "nesting_substrate_index_[SUBSTRATE].tif": {
                    "about": "Nesting substrate index for the given substrate",
                    "bands": {1: {"type": "ratio"}},
                },
                "relative_floral_abundance_index_[SEASON].tif": {
                    "about": "Floral abundance index in the given season",
                    "bands": {1: {"type": "ratio"}},
                },
                "reprojected_farm_vector.shp": {
                    "about": "Farm vector reprojected to the LULC projection",
                    "fields": {},
                    "geometries": spec_utils.POLYGONS,
                },
            },
        },
        "taskgraph_cache": spec_utils.TASKGRAPH_DIR,
    },
}

# Rebuild of InVEST Pollinator model
_INDEX_NODATA = -1.0

BASELINE_SCENARIO_LABEL = "baseline"

BASELINE_ARGS = [
    "landcover_biophysical_table_path",
    "landcover_raster_path",
]

# Parameter names
# These patterns are expected in the biophysical table
_NESTING_SUBSTRATE_PATTERN = "nesting_%s_availability_index"
_NESTING_SUBSTRATE_RE_PATTERN = _NESTING_SUBSTRATE_PATTERN % "([^_]+)"
_FLORAL_RESOURCES_AVAILABLE_PATTERN = "floral_resources_%s_index"
_FLORAL_RESOURCES_AVAILABLE_RE_PATTERN = _FLORAL_RESOURCES_AVAILABLE_PATTERN % "([^_]+)"
_EXPECTED_BIOPHYSICAL_HEADERS = [
    _NESTING_SUBSTRATE_RE_PATTERN,
    _FLORAL_RESOURCES_AVAILABLE_RE_PATTERN,
]

# These are patterns expected in the guilds table
_NESTING_SUITABILITY_PATTERN = "nesting_suitability_([^_]+)_index"
# replace with season
_FORAGING_ACTIVITY_PATTERN = "foraging_activity_%s_index"
_FORAGING_ACTIVITY_RE_PATTERN = _FORAGING_ACTIVITY_PATTERN % "([^_]+)"
_RELATIVE_SPECIES_ABUNDANCE_FIELD = "relative_abundance"
_ALPHA_HEADER = "alpha"
_EXPECTED_GUILD_HEADERS = [
    _NESTING_SUITABILITY_PATTERN,
    _FORAGING_ACTIVITY_RE_PATTERN,
    _ALPHA_HEADER,
    _RELATIVE_SPECIES_ABUNDANCE_FIELD,
]

# These patterns are expected in the biophysical table if calculating yield
_CROP_POLLINATOR_DEPENDENCE_PATTERN = "crop_pollinator_dependence_%s_index"
_CROP_POLLINATOR_DEPENDENCE_RE_PATTERN = _CROP_POLLINATOR_DEPENDENCE_PATTERN % "([^_]+)"
_HALF_SATURATION_FIELD = "half_saturation_coefficient"
_CROP_VALUE_FIELD = "value_per_hectare"
_EXPECTED_BIOPHYSICAL_YIELD_HEADERS = [
    _NESTING_SUBSTRATE_RE_PATTERN,
    _FLORAL_RESOURCES_AVAILABLE_RE_PATTERN,
    _CROP_POLLINATOR_DEPENDENCE_RE_PATTERN,
    _HALF_SATURATION_FIELD,
    _CROP_VALUE_FIELD,
]


# File patterns
# nesting substrate index raster replace (substrate, file_suffix)
_NESTING_SUBSTRATE_INDEX_FILEPATTERN = "nesting_substrate_index_%s%s.tif"
# habitat nesting index replaced by (species, file_suffix)
_HABITAT_NESTING_INDEX_FILE_PATTERN = "habitat_nesting_index_%s%s.tif"
# seasonal floral abundance index replaced by (season, file_suffix)
_SEASONAL_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN = (
    "seasonal_floral_abundance_index_%s%s.tif"
)
# seasonal forage (i.e. seasonal floral resources multiplied by species foraging activity)
# #replace (species, season, file_suffix)
_SEASONAL_FORAGE_INDEX_FILE_PATTERN = "seasonal_forage_index_%s_%s%s.tif"
# annual forage (i.e. sum of seasonal floral resources by species)
# #replace (species, file_suffix)
_ANNUAL_FORAGE_INDEX_FILE_PATTERN = "annual_forage_index_%s%s.tif"
# convolution of annual forage replace (species, file_suffix)
_FORAGING_EFFECTIVENESS_FILE_PATTERN = "foraging_effectiveness_%s%s.tif"
# convolution of habitat nesting replaced by (species, file_suffix)
_LOCAL_NESTING_DENSITY_FILE_PATTERN = "local_nesting_density_%s%s.tif"

# used to store the 2D decay kernel for a given distance replace
# (alpha, file suffix)
_KERNEL_FILE_PATTERN = "kernel_%f%s.tif"

# File patterns for species files
# landscape pollinator score for species replaced by (species, file_suffix)
_SPECIES_LANDSCAPE_SCORE_FILE_PATTERN = "pollinator_landscape_score_%s%s.tif"
# expected pollinator visitation index for species replaced by (species, file_suffix)
_SPECIES_POLLINATOR_VISITATION_FILE_PATTERN = "pollinator_visitation_index_%s%s.tif"
# marginal value for species replaced by (species, file_suffix)
_SPECIES_MARGINAL_VALUE_FILE_PATTERN = "marginal_value_%s%s.tif"
# marginal value replaced by file_suffix
_MARGINAL_VALUE_FILE_PATTERN = "marginal_value_%s.tif"

# FIle patterns for
# expected pollinator visitation index  replaced by (file_suffix)
_TOTAL_POLLINATOR_VISITATION_FILE_PATTERN = "total_pollinator_visitation_index_%s.tif"


# pollinator dependence raster replace (season, file_suffix)
_CROP_POLLINATOR_DEPENDENCE_FILE_PATTERN = "crop_pollinator_dependence_%s%s.tif"
# half saturation raster replace (file_suffix)
_HALF_SATURATION_FILE_PATTERN = "half_saturation_%s.tif"
# crop value raster replace (file_suffix)
_CROP_VALUE_FILE_PATTERN = "crop_value_%s.tif"
# seasonal crop yield raster replace (season, file_suffix)
_SEASONAL_YIELD_FILE_PATTERN = "seasonal_yield_%s%s.tif"
# annual crop yield raster replace (file_suffix)
_ANNUAL_YIELD_FILE_PATTERN = "annual_yield_%s.tif"
# realized value of crop yield raster replace (file_suffix)
_ANNUAL_YIELD_VALUE_FILE_PATTERN = "annual_yield_value_%s.tif"

# change in crop yield raster replace (season, file_suffix)
_DELTA_SEASONAL_YIELD_BY_VISITATION_FILE_PATTERN = (
    "d_seasonal_yield_d_pollinator_visitation_%s%s.tif"
)
# change in crop yield raster replace (file_suffix)
_DELTA_ANNUAL_YIELD_FILE_PATTERN = "d_annual_yield_d_pollinator_visitation_%s.tif"
# change in crop yield raster replace (season, file_suffix)
_DELTA_ANNUAL_YIELD_BY_LANDSCAPE_SCORE_FILE_PATTERN = (
    "d_annual_yield_d_pollinator_landscape_score_%s.tif"
)
# change in crop yield raster replace (season, file_suffix)
_DELTA_ANNUAL_YIELD_BY_HABITAT_FILE_PATTERN = "d_annual_yield_d_habitat_%s.tif"
# change in realized value of crop yield replace (season, file_suffix)
_DELTA_ANNUAL_YIELD_VALUE_FILE_PATTERN = "d_yield_value_%s.tif"


def execute(args):
    """InVEST Pollination Model.

    Args:
        args['workspace_dir'] (string): a path to the output workspace folder.
            Will overwrite any files that exist if the path already exists.
        args['results_suffix'] (string): string appended to each output
            file path.
        args['landcover_raster_path'] (string): file path to a landcover
            raster.
        args['guild_table_path'] (string): file path to a table indicating
            the bee species to analyze in this model run.  Table headers
            must include:
                * 'species': a bee species whose column string names will
                    be referred to in other tables and the model will output
                    analyses per species.
                * one or more columns matching _NESTING_SUITABILITY_PATTERN
                    with values in the range [0.0, 1.0] indicating the
                    suitability of the given species to nest in a particular
                    substrate.
                * one or more columns matching _FORAGING_ACTIVITY_RE_PATTERN
                    with values in the range [0.0, 1.0] indicating the
                    relative level of foraging activity for that species
                    during a particular season.
                * _ALPHA_HEADER the sigma average flight distance of that bee
                    species in meters.
                * 'relative_abundance': a weight indicating the relative
                    abundance of the particular species with respect to the
                    sum of all relative abundance weights in the table.

        args['landcover_biophysical_table_path'] (string): path to a table
            mapping landcover codes in `args['landcover_path']` to indexes of
            nesting availability for each nesting substrate referenced in
            guilds table as well as indexes of abundance of floral resources
            on that landcover type per season in the bee activity columns of
            the guild table.

            All indexes are in the range [0.0, 1.0].

            Columns in the table must be at least
                * 'lucode': representing all the unique landcover codes in
                    the raster ast `args['landcover_path']`
                * For every nesting matching _NESTING_SUITABILITY_PATTERN
                  in the guild stable, a column matching the pattern in
                  `_LANDCOVER_NESTING_INDEX_HEADER`.
                * For every season matching _FORAGING_ACTIVITY_RE_PATTERN
                  in the guilds table, a column matching
                  the pattern in `_LANDCOVER_FLORAL_RESOURCES_INDEX_HEADER`.
        args['scenario_labels_list'] (list): (optional) list of strings appended
            to each output file path for each scenario. Will be replaced by
            'scenario1', 'scenario2', etc. if not provided.
        args['scenario_landcover_biophysical_table_path_list'] (list): (optional)
            list of file paths to alternative biophysical parameter tables. Must
            be identical in structure to `args['landcover_biophysical_table_path']`.
        args['scenario_landcover_raster_path_list'] (list): (optional) list
            of file paths to alternative landcover scenario rasters. Should be
            identical in resolution and extent to `args['landcover_raster_path']`.
        args['calculate_yield'] (bool): (optional) If True, calculate the marginal
            change in pollinator-dependent agriculture.
        args['farm_vector_path'] (string): (optional) path to a vector of farm locations.
        args['aggregate_size'] (int): (optional) A rescaling factor used to
            aggregate all rasters to ease convolution calculations.
        args['n_workers'] (int): (optional) The number of worker processes to
            use for processing this model.  If omitted, computation will take
            place in the current process.

    Returns:
        None
    """
    LOGGER.info("Starting NCI Pollinator model")

    # create initial working directories and determine file suffixes
    output_dir = Path(args["workspace_dir"])
    intermediate_output_dir = output_dir / "intermediate_outputs"
    species_season_dir = intermediate_output_dir / "species_season_outputs"
    species_dir = intermediate_output_dir / "species_outputs"
    season_dir = intermediate_output_dir / "season_outputs"

    work_token_dir = intermediate_output_dir / "_taskgraph_working_dir"

    utils.make_directories(
        [
            output_dir,
            intermediate_output_dir,
            species_season_dir,
            season_dir,
            work_token_dir,
        ]
    )
    file_suffix = utils.make_suffix_string(args, "results_suffix")

    landcover_raster_info = pygeoprocessing.get_raster_info(
        str(args["landcover_raster_path"])
    )

    try:
        n_workers = int(args["n_workers"])
    except (KeyError, ValueError, TypeError):
        # KeyError when n_workers is not present in args
        # ValueError when n_workers is an empty string.
        # TypeError when n_workers is None.
        n_workers = -1  # Synchronous mode.
    task_graph = taskgraph.TaskGraph(work_token_dir, n_workers)

    # Create scenario iterator (even if there is only one)
    scenario_labels_list, scenario_args_dict = nature.scenario_labeling(
        args,
        BASELINE_ARGS,
        baseline_scenario_label=BASELINE_SCENARIO_LABEL,
    )

    # Test if required yield parameters are present, if yield is to be calculated
    # TODO figure out if we need a farm_vector_path
    # if "calculate_yield" in args and args["calculate_yield"]:
    #     # Check if farm vector is present
    #     try:
    #         args["farm_vector_path"]
    #     except:
    #         LOGGER.error(f"Must provide a farm vector path if calculating yield.")

    # Create taskgraph task dictionaries
    landcover_substrate_index_task_map = {name: {} for name in scenario_labels_list}
    habitat_nesting_task_map = {name: {} for name in scenario_labels_list}
    seasonal_floral_abundance_task_map = {name: {} for name in scenario_labels_list}
    seasonal_forage_index_task_map = {name: {} for name in scenario_labels_list}
    annual_forage_index_task_map = {name: {} for name in scenario_labels_list}
    warp_rasters_task_map = {
        "habitat_nesting": {name: {} for name in scenario_labels_list},
        "annual_forage": {name: {} for name in scenario_labels_list},
    }
    species_foraging_effectiveness_index_task_map = {
        name: {} for name in scenario_labels_list
    }
    species_landscape_score_task_map = {name: {} for name in scenario_labels_list}

    species_pollinator_visitation_task_map = {name: {} for name in scenario_labels_list}

    species_marginal_value_task_map = {name: {} for name in scenario_labels_list}
    marginal_value_task_map = {name: {} for name in scenario_labels_list}
    total_pollinator_visitation_task_map = {name: {} for name in scenario_labels_list}

    # parse out the scenario variables from a complicated set of two tables
    # and possibly a farm polygon.  This function will also raise an exception
    # if any of the inputs are malformed.
    scenario_filepath_list = [
        "nesting_substrate_index_path",
        "habitat_nesting_index_path",
        "seasonal_floral_abundance_index_path",
        "seasonal_forage_index_path",
        "annual_forage_index_path",
        "foraging_effectiveness_index_path",
        "species_landscape_score_path",
        "species_pollinator_visitation_index_path",
        "total_pollinator_visitation_index_path",
        "d_pollinator_species_landscape_score_path",
        "d_pollinator_landscape_score_path",
    ]
    scenario_variables = _parse_scenario_variables(
        args, scenario_labels_list, scenario_args_dict, scenario_filepath_list
    )

    # Iterate through scenarios (even if there is only one)
    for scenario_name in scenario_labels_list:
        # Create scenario suffix
        file_suffix = nature.make_scenario_suffix(args["results_suffix"], scenario_name)
        # Reclass LULC to Nesting (by substrate)
        # calculate nesting_substrate_index[substrate] substrate maps
        # N(x, n) = ln(l(x), n)

        reclass_error_details = {
            "raster_name": "LULC",
            "column_name": "lucode",
            "table_name": "Biophysical",
        }
        for substrate in scenario_variables["substrate_list"]:
            LOGGER.info(
                f"{scenario_name} | Creating Nesting Substrate raster for substrate: {substrate}"
            )
            nesting_substrate_index_path = intermediate_output_dir / (
                _NESTING_SUBSTRATE_INDEX_FILEPATTERN % (substrate, file_suffix)
            )

            landcover_substrate_index_task_map[scenario_name][
                substrate
            ] = task_graph.add_task(
                task_name=f"reclassify_to_substrate_{substrate}_{scenario_name}",
                func=utils.reclassify_raster,
                args=(
                    [
                        (
                            str(
                                scenario_args_dict["landcover_raster_path"][
                                    scenario_name
                                ]
                            ),
                            1,
                        ),
                        scenario_variables["landcover_substrate_index"][scenario_name][
                            substrate
                        ],
                        str(nesting_substrate_index_path),
                        gdal.GDT_Float32,
                        _INDEX_NODATA,
                        reclass_error_details,
                    ]
                ),
                target_path_list=[str(nesting_substrate_index_path)],
            )

            scenario_variables["nesting_substrate_index_path"][scenario_name][
                substrate
            ] = nesting_substrate_index_path

        # Calculate Habitat Nesting (based on species and substrate)
        # calculate habitat_nesting_index[species] HN(x, s) = max_n(N(x, n) ns(s,n))
        for species in scenario_variables["species_list"]:
            LOGGER.info(
                f"{scenario_name} | Creating Habitat Nesting Index raster for species: {species}"
            )

            habitat_nesting_index_path = species_dir / (
                _HABITAT_NESTING_INDEX_FILE_PATTERN % (species, file_suffix)
            )

            calculate_habitat_nesting_index_op = _CalculateHabitatNestingIndex(
                scenario_variables["nesting_substrate_index_path"][scenario_name],
                scenario_variables["species_substrate_index"][species],
                str(habitat_nesting_index_path),
            )

            habitat_nesting_task_map[scenario_name][species] = task_graph.add_task(
                task_name=f"calculate_habitat_nesting_{species}_{scenario_name}",
                func=calculate_habitat_nesting_index_op,
                dependent_task_list=landcover_substrate_index_task_map[
                    scenario_name
                ].values(),
                target_path_list=[str(habitat_nesting_index_path)],
            )

            scenario_variables["habitat_nesting_index_path"][scenario_name][
                species
            ] = habitat_nesting_index_path

        # Calculate Seasonal Floral Abundance (based on season)
        # calculate seasonal_floral_abundance_index[season] per season RA(l(x), j)
        reclass_error_details = {
            "raster_name": "LULC",
            "column_name": "lucode",
            "table_name": "Biophysical",
        }
        for season in scenario_variables["season_list"]:
            LOGGER.info(
                f"{scenario_name} | Creating Seasonal Floral Abundance raster for season: {season}"
            )
            seasonal_floral_abundance_index_path = season_dir / (
                _SEASONAL_FLORAL_ABUNDANCE_INDEX_FILE_PATTERN % (season, file_suffix)
            )

            seasonal_floral_abundance_task_map[scenario_name][
                season
            ] = task_graph.add_task(
                task_name=f"reclassify_to_floral_abundance_{season}_{scenario_name}",
                func=utils.reclassify_raster,
                args=(
                    [
                        (
                            str(
                                scenario_args_dict["landcover_raster_path"][
                                    scenario_name
                                ]
                            ),
                            1,
                        ),
                        scenario_variables["landcover_floral_resources"][scenario_name][
                            season
                        ],
                        str(seasonal_floral_abundance_index_path),
                        gdal.GDT_Float32,
                        _INDEX_NODATA,
                        reclass_error_details,
                    ]
                ),
                target_path_list=[str(seasonal_floral_abundance_index_path)],
            )

            scenario_variables["seasonal_floral_abundance_index_path"][scenario_name][
                season
            ] = seasonal_floral_abundance_index_path

        # calculate season- and species-dependent forage
        for species in scenario_variables["species_list"]:
            # calculate seasonal_forage_species_season = RA(l(x),j)*fa(s,j)
            for season in scenario_variables["season_list"]:
                LOGGER.info(
                    f"{scenario_name} | Creating Seasonal Forage raster for species, season: {species}, {season}"
                )
                seasonal_forage_index_path = species_season_dir / (
                    _SEASONAL_FORAGE_INDEX_FILE_PATTERN % (species, season, file_suffix)
                )
                seasonal_floral_abundance_path = scenario_variables[
                    "seasonal_floral_abundance_index_path"
                ][scenario_name][season]
                multiply_by_scalar_op = rasterops.MultiplyRasterByScalar(
                    scenario_variables["species_foraging_activity"][(species, season)],
                    _INDEX_NODATA,
                )
                seasonal_forage_index_task_map[scenario_name][
                    (species, season)
                ] = task_graph.add_task(
                    task_name=f"calculate_seasonal_forage_{species}_{season}_{scenario_name}",
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        [
                            [(str(seasonal_floral_abundance_path), 1)],
                            multiply_by_scalar_op,
                            str(seasonal_forage_index_path),
                            gdal.GDT_Float32,
                            _INDEX_NODATA,
                        ]
                    ),
                    dependent_task_list=[
                        seasonal_floral_abundance_task_map[scenario_name][season]
                    ],
                    target_path_list=[str(seasonal_forage_index_path)],
                )
                scenario_variables["seasonal_forage_index_path"][scenario_name][
                    (species, season)
                ] = seasonal_forage_index_path

            # calculate annual_forage_species: FE(x, s) = sum_j [RA(l(x), j) * fa(s, j)]
            seasonal_forage_path_band_list = [
                (
                    str(
                        scenario_variables["seasonal_forage_index_path"][scenario_name][
                            (species, season)
                        ]
                    ),
                    1,
                )
                for season in scenario_variables["season_list"]
            ]
            annual_forage_index_path = species_dir / (
                _ANNUAL_FORAGE_INDEX_FILE_PATTERN % (species, file_suffix)
            )

            annual_forage_index_task_map[scenario_name][species] = task_graph.add_task(
                task_name=f"annual_forage_{species}",
                func=pygeoprocessing.raster_calculator,
                args=(
                    seasonal_forage_path_band_list,
                    sum_raster_op,
                    str(annual_forage_index_path),
                    gdal.GDT_Float32,
                    _INDEX_NODATA,
                ),
                target_path_list=[str(annual_forage_index_path)],
                dependent_task_list=[
                    seasonal_forage_index_task_map[scenario_name][(species, season)]
                    for season in scenario_variables["season_list"]
                ],
            )

            scenario_variables["annual_forage_index_path"][scenario_name][
                species
            ] = annual_forage_index_path

        # Aggregate raster data if necessary
        if args["aggregate_size"] is not None:
            LOGGER.info(
                f"{scenario_name} | Aggregating forage and nesting resource rasters"
            )
            landcover_pixel_size_tuple = tuple(
                args["aggregate_size"] * x for x in landcover_raster_info["pixel_size"]
            )
            for species in scenario_variables["species_list"]:
                h_nesting = scenario_variables["habitat_nesting_index_path"][
                    scenario_name
                ][species]
                scenario_variables["habitat_nesting_index_path"][scenario_name][
                    species
                ] = h_nesting.replace(".tif", "_aggregated.tif")
                warp_rasters_task_map["habitat_nesting"][scenario_name][
                    species
                ] = task_graph.add_task(
                    task_name=f"aggregate_habitat_nesting_raster_{species}_{scenario_name}",
                    func=pygeoprocessing.warp_raster,
                    args=(
                        [
                            str(h_nesting),
                            landcover_pixel_size_tuple,
                            str(
                                scenario_variables["habitat_nesting_index_path"][
                                    scenario_name
                                ][species]
                            ),
                        ]
                    ),
                    kwargs={
                        "resample_method": "average",
                        "target_bb": landcover_raster_info["bounding_box"],
                    },
                    dependent_task_list=[
                        habitat_nesting_task_map[scenario_name][species]
                    ],
                    target_path_list=[
                        scenario_variables["habitat_nesting_index_path"][scenario_name][
                            species
                        ]
                    ],
                )

                # Aggregate annual forage index
                forage_index = scenario_variables["annual_forage_index_path"][
                    scenario_name
                ][species]
                scenario_variables["annual_forage_index_path"][scenario_name][
                    species
                ] = forage_index.replace(".tif", "_aggregated.tif")
                warp_rasters_task_map["annual_forage"][scenario_name][
                    species
                ] = task_graph.add_task(
                    task_name=f"aggregate_annual_forage_raster_{species}_{scenario_name}",
                    func=pygeoprocessing.warp_raster,
                    args=(
                        [
                            str(forage_index),
                            landcover_pixel_size_tuple,
                            str(
                                scenario_variables["annual_forage_index_path"][
                                    scenario_name
                                ][species]
                            ),
                        ]
                    ),
                    kwargs={
                        "resample_method": "average",
                        "target_bb": landcover_raster_info["bounding_box"],
                    },
                    dependent_task_list=[
                        annual_forage_index_task_map[scenario_name][species]
                    ],
                    target_path_list=[
                        scenario_variables["annual_forage_index_path"][scenario_name][
                            [species]
                        ]
                    ],
                )

        else:
            warp_rasters_task_map["habitat_nesting"][
                scenario_name
            ] = habitat_nesting_task_map[scenario_name]
            warp_rasters_task_map["annual_forage"][
                scenario_name
            ] = annual_forage_index_task_map[scenario_name]
            landcover_pixel_size_tuple = landcover_raster_info["pixel_size"]

        # Iterate through species and calculate marginal values and baseline landscape score and visitation
        for species in scenario_variables["species_list"]:
            # Define pixel size
            try:
                landcover_mean_pixel_size = utils.mean_pixel_size_and_area(
                    landcover_pixel_size_tuple
                )[0]
            except ValueError:
                landcover_mean_pixel_size = np.min(
                    np.absolute(landcover_pixel_size_tuple)
                )
                LOGGER.debug(
                    "Land Cover Raster has unequal x, y pixel sizes: %s. Using"
                    "%s as the mean pixel size."
                    % (landcover_pixel_size_tuple, landcover_mean_pixel_size)
                )
            # create a convolution kernel for the species flight range
            LOGGER.info(
                f"{scenario_name} | Creating convolution kernel for species: {species}"
            )
            alpha = (
                scenario_variables["alpha_value"][species] / landcover_mean_pixel_size
            )
            kernel_path = intermediate_output_dir / (
                _KERNEL_FILE_PATTERN % (alpha, file_suffix)
            )

            alpha_kernel_raster_task = task_graph.add_task(
                task_name=f"decay_kernel_raster_{alpha}",
                func=utils.exponential_decay_kernel_raster,
                args=([alpha, str(kernel_path)]),
                target_path_list=[str(kernel_path)],
                dependent_task_list=(
                    [warp_rasters_task_map["habitat_nesting"][scenario_name][species]]
                    + [warp_rasters_task_map["annual_forage"][scenario_name][species]]
                ),
            )

            if scenario_name == BASELINE_SCENARIO_LABEL:
                # convolve HN with alpha_s
                LOGGER.info(
                    f"{scenario_name} | Calculating Local Nesting Density convolution for species: {species}"
                )
                baseline_local_nesting_density_path = intermediate_output_dir / (
                    _LOCAL_NESTING_DENSITY_FILE_PATTERN % (species, file_suffix)
                )

                baseline_local_nesting_density_task = task_graph.add_task(
                    task_name=f"convolve_nesting_{species}_{scenario_name}",
                    func=pygeoprocessing.convolve_2d,
                    args=(
                        [
                            (
                                str(
                                    scenario_variables["habitat_nesting_index_path"][
                                        scenario_name
                                    ][species]
                                ),
                                1,
                            ),
                            (str(kernel_path), 1),
                            str(baseline_local_nesting_density_path),
                        ]
                    ),
                    kwargs={
                        "ignore_nodata_and_edges": True,
                        "mask_nodata": True,
                        "normalize_kernel": False,
                    },
                    dependent_task_list=[
                        alpha_kernel_raster_task,
                        habitat_nesting_task_map[scenario_name][species],
                    ],
                    target_path_list=[str(baseline_local_nesting_density_path)],
                )

                # convolve annual forage with alpha_s
                LOGGER.info(
                    f"{scenario_name} | Calculating Foraging Effectiveness convolution for species: {species}"
                )
                foraging_effectiveness_index_path = species_dir / (
                    _FORAGING_EFFECTIVENESS_FILE_PATTERN % (species, file_suffix)
                )

                species_foraging_effectiveness_index_task_map[scenario_name][
                    species
                ] = task_graph.add_task(
                    task_name=f"convolve_annual_forage_{species}_{scenario_name}",
                    func=pygeoprocessing.convolve_2d,
                    args=(
                        [
                            (
                                str(
                                    scenario_variables["annual_forage_index_path"][
                                        scenario_name
                                    ][species]
                                ),
                                1,
                            ),
                            (str(kernel_path), 1),
                            str(foraging_effectiveness_index_path),
                        ]
                    ),
                    kwargs={
                        "ignore_nodata_and_edges": True,
                        "mask_nodata": True,
                        "normalize_kernel": False,
                    },
                    dependent_task_list=[
                        alpha_kernel_raster_task,
                    ],
                    target_path_list=[str(foraging_effectiveness_index_path)],
                )

                scenario_variables["foraging_effectiveness_index_path"][scenario_name][
                    species
                ] = foraging_effectiveness_index_path

                # calculate pollinator landscape score
                LOGGER.info(
                    f"{scenario_name} | Calculating Pollinator Landscape Score for species: {species}"
                )
                species_landscape_score_path = species_dir / (
                    _SPECIES_LANDSCAPE_SCORE_FILE_PATTERN % (species, file_suffix)
                )

                species_landscape_score_task_map[scenario_name][
                    species
                ] = task_graph.add_task(
                    task_name=f"calculate_pollinator_landscape_score_{species}_{scenario_name}",
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        [
                            (
                                str(
                                    scenario_variables["habitat_nesting_index_path"][
                                        BASELINE_SCENARIO_LABEL
                                    ][species]
                                ),
                                1,
                            ),
                            (str(foraging_effectiveness_index_path), 1),
                        ],
                        multiply_rasters_op,
                        str(species_landscape_score_path),
                        gdal.GDT_Float32,
                        _INDEX_NODATA,
                    ),
                    dependent_task_list=[
                        warp_rasters_task_map["habitat_nesting"][scenario_name][
                            species
                        ],
                        species_foraging_effectiveness_index_task_map[scenario_name][
                            species
                        ],
                    ],
                    target_path_list=[str(species_landscape_score_path)],
                )

                scenario_variables["species_landscape_score_path"][scenario_name][
                    species
                ] = species_landscape_score_path

                # Calculate pollinator visitation by convolving landscape score by alpha
                # QUESTION should we weight this visitation by relative abundance? This step is done within the TOTAL_VISITATION
                # QUESTION step already but it might be clearer if we do it here, so the rasters will clearly sum together.
                LOGGER.info(
                    f"{scenario_name} | Calculating Pollinator Visitation convolution for species: {species}"
                )
                species_pollinator_visitation_index_path = species_dir / (
                    _SPECIES_POLLINATOR_VISITATION_FILE_PATTERN % (species, file_suffix)
                )

                species_pollinator_visitation_task_map[scenario_name][
                    species
                ] = task_graph.add_task(
                    task_name=f"convolve_species_landscape_score_{species}_{scenario_name}",
                    func=pygeoprocessing.convolve_2d,
                    args=(
                        [
                            (
                                str(
                                    scenario_variables["species_landscape_score_path"][
                                        scenario_name
                                    ][species]
                                ),
                                1,
                            ),
                            (str(kernel_path), 1),
                            str(species_pollinator_visitation_index_path),
                        ]
                    ),
                    kwargs={
                        "ignore_nodata_and_edges": True,
                        "mask_nodata": True,
                        "normalize_kernel": False,
                    },
                    dependent_task_list=[
                        species_landscape_score_task_map[scenario_name][species],
                    ],
                    target_path_list=[str(species_pollinator_visitation_index_path)],
                )

                scenario_variables["species_pollinator_visitation_index_path"][
                    scenario_name
                ][species] = species_pollinator_visitation_index_path

            # Calculate marginal change in pollinator landscape score (d_pollinator_species_landscape_score_path) per species if the scenario is not the baseline
            # d_pollinator_landscape_score represents the change in pollinator landscape score
            # d_habitat is a shorthand for the change in floral and nesting values on the landscape
            else:
                # calculate species marginal value function
                LOGGER.info(
                    f"{scenario_name} | Calculating biophysical Marginal Value raster for species: {species}"
                )
                d_pollinator_species_landscape_score_path = species_dir / (
                    _SPECIES_MARGINAL_VALUE_FILE_PATTERN % (species, file_suffix)
                )

                species_marginal_value_task_map[scenario_name] = task_graph.add_task(
                    task_name=f"calculate_marginal_value_{species}_{scenario_name}",
                    func=d_pollinator_landscape_score_task_op,
                    args=[
                        kernel_path,
                        scenario_variables["species_abundance"][species],
                        baseline_local_nesting_density_path,
                        scenario_variables["foraging_effectiveness_index_path"][
                            BASELINE_SCENARIO_LABEL
                        ][species],
                        scenario_variables["annual_forage_index_path"][
                            BASELINE_SCENARIO_LABEL
                        ][species],
                        scenario_variables["annual_forage_index_path"][scenario_name][
                            species
                        ],
                        scenario_variables["habitat_nesting_index_path"][
                            BASELINE_SCENARIO_LABEL
                        ][species],
                        scenario_variables["habitat_nesting_index_path"][scenario_name][
                            species
                        ],
                        d_pollinator_species_landscape_score_path,
                    ],
                    dependent_task_list=[
                        baseline_local_nesting_density_task,
                        species_foraging_effectiveness_index_task_map[
                            BASELINE_SCENARIO_LABEL
                        ][species],
                        warp_rasters_task_map["annual_forage"][BASELINE_SCENARIO_LABEL][
                            species
                        ],
                        warp_rasters_task_map["annual_forage"][scenario_name][species],
                        warp_rasters_task_map["habitat_nesting"][
                            BASELINE_SCENARIO_LABEL
                        ][species],
                        warp_rasters_task_map["habitat_nesting"][scenario_name][
                            species
                        ],
                    ],
                    target_path_list=[str(d_pollinator_species_landscape_score_path)],
                )

                scenario_variables["d_pollinator_species_landscape_score_path"][
                    scenario_name
                ][species] = d_pollinator_species_landscape_score_path

        # [x] TODO total pollinator visitation
        if scenario_name == BASELINE_SCENARIO_LABEL:
            # calculate total pollinator visitation by summing all species visitation rates
            # (multiplied by their relative abundance)
            LOGGER.info(f"{scenario_name} | Sum Pollinator Visitation for all species")
            total_pollinator_visitation_index_path = output_dir / (
                _TOTAL_POLLINATOR_VISITATION_FILE_PATTERN % file_suffix
            )

            species_pollinator_visitation_index_path_band_list = []
            species_abundance_list = []
            for species in scenario_variables["species_list"]:
                species_pollinator_visitation_index_path_band_list.append(
                    (
                        str(
                            scenario_variables[
                                "species_pollinator_visitation_index_path"
                            ][scenario_name][species]
                        ),
                        1,
                    )
                )
                species_abundance_list.append(
                    scenario_variables["species_abundance"][species]
                )

            sum_by_scalar_raster_op = rasterops.SumRastersByScalar(
                species_abundance_list, _INDEX_NODATA
            )
            # TODO Change to a sum rather than sum by scalar (once we have relative abundance calculated earlier)
            total_pollinator_visitation_task_map[scenario_name] = task_graph.add_task(
                task_name=f"calculate_total_pollinator_visitation__{scenario_name}",
                func=pygeoprocessing.raster_calculator,
                args=(
                    species_pollinator_visitation_index_path_band_list,
                    sum_by_scalar_raster_op,
                    str(total_pollinator_visitation_index_path),
                    gdal.GDT_Float32,
                    _INDEX_NODATA,
                ),
                dependent_task_list=[
                    species_pollinator_visitation_task_map[scenario_name][species]
                    for species in scenario_variables["species_list"]
                ],
                target_path_list=[str(total_pollinator_visitation_index_path)],
            )

            scenario_variables["total_pollinator_visitation_index_path"][
                scenario_name
            ] = total_pollinator_visitation_index_path
        else:
            # TODO Sum species marginal values??? How will this math work?
            LOGGER.info(
                f"{scenario_name} | Sum biophysical marginal values across species"
            )

            d_pollinator_species_landscape_score_path_band_list = [
                (
                    str(
                        scenario_variables["d_pollinator_species_landscape_score_path"][
                            scenario_name
                        ][species]
                    ),
                    1,
                )
                for species in scenario_variables["species_list"]
            ]

            d_pollinator_landscape_score_path = output_dir / (
                _MARGINAL_VALUE_FILE_PATTERN % file_suffix
            )

            marginal_value_task_map[scenario_name] = task_graph.add_task(
                task_name=f"calculate_marginal_value_{scenario_name}",
                func=pygeoprocessing.raster_calculator,
                args=[
                    d_pollinator_species_landscape_score_path_band_list,
                    sum_raster_op,
                    str(d_pollinator_landscape_score_path),
                    gdal.GDT_Float32,
                    _INDEX_NODATA,
                ],
                dependent_task_list=[species_marginal_value_task_map[scenario_name]],
                target_path_list=[str(d_pollinator_landscape_score_path)],
            )

            scenario_variables["d_pollinator_landscape_score_path"][
                scenario_name
            ] = d_pollinator_landscape_score_path

        if "calculate_yield" in args and args["calculate_yield"]:
            # Create additional scenario_variable dictionaries
            for scenario_variable_list in [
                "crop_pollinator_dependence_raster_path",
                "half_saturation_coefficient_raster_path",
                "crop_value_raster_path",
                "seasonal_yield_raster_path",
                "annual_yield_raster_path",
                "annual_yield_value_raster_path",
                "delta_seasonal_yield_raster_path",
                "delta_annual_yield_raster_path",
                "delta_annual_yield_convolution_raster_path",
                "delta_annual_yield_value_raster_path",
            ]:
                scenario_variables[scenario_variable_list] = {
                    name: {} for name in scenario_labels_list
                }

            # Create taskgraph task dictionaries
            crop_pollinator_dependence_tasks = {
                name: {} for name in scenario_labels_list
            }
            half_saturation_tasks = {name: {} for name in scenario_labels_list}
            crop_value_tasks = {name: {} for name in scenario_labels_list}
            seasonal_yield_tasks = {name: {} for name in scenario_labels_list}
            annual_yield_tasks = {name: {} for name in scenario_labels_list}
            annual_yield_value_tasks = {name: {} for name in scenario_labels_list}
            delta_seasonal_yield_tasks = {name: {} for name in scenario_labels_list}
            delta_annual_yield_tasks = {name: {} for name in scenario_labels_list}
            delta_annual_yield_convolution_tasks = {
                name: {} for name in scenario_labels_list
            }
            delta_annual_yield_habitat_tasks = {
                name: {} for name in scenario_labels_list
            }
            delta_annual_yield_value_tasks = {name: {} for name in scenario_labels_list}

            # Reclassify raster to crop pollinator dependence
            for season in scenario_variables["season_list"]:
                LOGGER.info(
                    f"{scenario_name} | Creating Crop Pollinator Dependence raster for season: {season}"
                )
                crop_pollinator_dependence_raster_path = season_dir / (
                    _CROP_POLLINATOR_DEPENDENCE_FILE_PATTERN % (season, file_suffix)
                )

                crop_pollinator_dependence_tasks[scenario_name][
                    season
                ] = task_graph.add_task(
                    task_name=f"reclassify_to_crop_pollinator_dependence_{season}_{scenario_name}",
                    func=utils.reclassify_raster,
                    args=(
                        [
                            (
                                str(
                                    scenario_args_dict["landcover_raster_path"][
                                        scenario_name
                                    ]
                                ),
                                1,
                            ),
                            scenario_variables["crop_pollinator_dependence"][
                                scenario_name
                            ][season],
                            str(crop_pollinator_dependence_raster_path),
                            gdal.GDT_Float32,
                            _INDEX_NODATA,
                            reclass_error_details,
                        ]
                    ),
                    target_path_list=[str(crop_pollinator_dependence_raster_path)],
                )

                scenario_variables["crop_pollinator_dependence_raster_path"][
                    scenario_name
                ][season] = crop_pollinator_dependence_raster_path

            # Reclassify raster to half saturation
            LOGGER.info(
                f"{scenario_name} | Creating Half Saturation Coefficient raster"
            )
            half_saturation_coefficient_raster_path = intermediate_output_dir / (
                _HALF_SATURATION_FILE_PATTERN % file_suffix
            )

            half_saturation_tasks[scenario_name] = task_graph.add_task(
                task_name=f"reclassify_to_half_saturation_{scenario_name}",
                func=utils.reclassify_raster,
                args=(
                    (
                        str(scenario_args_dict["landcover_raster_path"][scenario_name]),
                        1,
                    ),
                    scenario_variables["half_saturation_coefficient"][scenario_name],
                    str(half_saturation_coefficient_raster_path),
                    gdal.GDT_Float32,
                    _INDEX_NODATA,
                    reclass_error_details,
                ),
                target_path_list=[str(half_saturation_coefficient_raster_path)],
            )

            scenario_variables["half_saturation_coefficient_raster_path"][
                scenario_name
            ] = half_saturation_coefficient_raster_path

            # Convert crop value per hectare to per pixel
            pixel_area = abs(np.prod(landcover_pixel_size_tuple))
            crop_value_per_pixel_dict = {
                lulc_id: crop_value * pixel_area / 10**4
                for lulc_id, crop_value in scenario_variables["crop_value"][
                    scenario_name
                ].items()
            }

            # Reclassify raster to crop value
            LOGGER.info(f"{scenario_name} | Creating Crop Value raster")
            crop_value_raster_path = intermediate_output_dir / (
                _CROP_VALUE_FILE_PATTERN % file_suffix
            )

            crop_value_tasks[scenario_name] = task_graph.add_task(
                task_name=f"reclassify_to_crop_value_{scenario_name}",
                func=utils.reclassify_raster,
                args=(
                    (
                        str(scenario_args_dict["landcover_raster_path"][scenario_name]),
                        1,
                    ),
                    crop_value_per_pixel_dict,
                    str(crop_value_raster_path),
                    gdal.GDT_Float32,
                    _INDEX_NODATA,
                    reclass_error_details,
                ),
                target_path_list=[str(crop_value_raster_path)],
            )

            scenario_variables["crop_value_raster_path"][
                scenario_name
            ] = crop_value_raster_path

            if scenario_name == BASELINE_SCENARIO_LABEL:
                for season in scenario_variables["season_list"]:
                    LOGGER.info(
                        f"{scenario_name} | Creating Crop Yield raster for season: {season}"
                    )
                    seasonal_yield_raster_path = season_dir / (
                        _SEASONAL_YIELD_FILE_PATTERN % (season, file_suffix)
                    )

                    seasonal_yield_tasks[scenario_name][season] = task_graph.add_task(
                        task_name=f"calculate_baseline_yield_{season}_{scenario_name}",
                        func=pygeoprocessing.raster_calculator,
                        args=(
                            [
                                (
                                    str(
                                        scenario_variables[
                                            "crop_pollinator_dependence_raster_path"
                                        ][scenario_name][season]
                                    ),
                                    1,
                                ),
                                (
                                    str(
                                        scenario_variables[
                                            "half_saturation_coefficient_raster_path"
                                        ][scenario_name]
                                    ),
                                    1,
                                ),
                                (
                                    str(
                                        scenario_variables[
                                            "total_pollinator_visitation_index_path"
                                        ][
                                            scenario_name
                                        ]  # NOTE this is annual visitation, not seasonal
                                    ),
                                    1,
                                ),
                            ],
                            yield_op,
                            str(seasonal_yield_raster_path),
                            gdal.GDT_Float32,
                            _INDEX_NODATA,
                        ),
                        dependent_task_list=[
                            crop_pollinator_dependence_tasks[scenario_name][season]
                            for season in scenario_variables["season_list"]
                        ]
                        + [
                            half_saturation_tasks[scenario_name],
                            total_pollinator_visitation_task_map[scenario_name],
                        ],
                        target_path_list=[str(seasonal_yield_raster_path)],
                    )

                    scenario_variables["seasonal_yield_raster_path"][scenario_name][
                        season
                    ] = seasonal_yield_raster_path

                LOGGER.info(f"{scenario_name} | Creating Annual Crop Yield raster")
                # calculate annual yield from seasonal yields
                # QUESTION should we ensure this value does not exceed 1? It's possible that crops have multiple seasons where
                # QUESTION pollinators contribute to annual yield, so when summing we may exceed 1.
                # QUESTION Solution is either Max(yield) or Max(yield, 1)
                seasonal_yield_raster_path_band_list = [
                    (
                        str(
                            scenario_variables["seasonal_yield_raster_path"][
                                scenario_name
                            ][season]
                        ),
                        1,
                    )
                    for season in scenario_variables["season_list"]
                ]
                annual_yield_raster_path = output_dir / (
                    _ANNUAL_YIELD_FILE_PATTERN % file_suffix
                )

                annual_yield_tasks[scenario_name] = task_graph.add_task(
                    task_name=f"calculate_annual_crop_yield_{scenario_name}",
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        seasonal_yield_raster_path_band_list,
                        sum_raster_op,  # QUESTION switch with sum_raster_cap_one_op
                        str(annual_yield_raster_path),
                        gdal.GDT_Float32,
                        _INDEX_NODATA,
                    ),
                    target_path_list=[str(annual_yield_raster_path)],
                    dependent_task_list=[
                        seasonal_yield_tasks[scenario_name][season]
                        for season in scenario_variables["season_list"]
                    ],
                )

                scenario_variables["annual_yield_raster_path"][
                    scenario_name
                ] = annual_yield_raster_path

                # calculate annual crop yield value
                LOGGER.info(
                    f"{scenario_name} | Calculating Annual Crop Yield Value raster"
                )
                annual_yield_value_raster_path = output_dir / (
                    _ANNUAL_YIELD_VALUE_FILE_PATTERN % file_suffix
                )

                annual_yield_value_tasks[scenario_name] = task_graph.add_task(
                    task_name=f"calculate_annual_yield_value_{scenario_name}",
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        [
                            (
                                str(
                                    scenario_variables["annual_yield_raster_path"][
                                        scenario_name
                                    ]
                                ),
                                1,
                            ),
                            (
                                str(
                                    scenario_variables["crop_value_raster_path"][
                                        scenario_name
                                    ]
                                ),
                                1,
                            ),
                        ],
                        multiply_rasters_op,
                        str(annual_yield_value_raster_path),
                        gdal.GDT_Float32,
                        _INDEX_NODATA,
                    ),
                    dependent_task_list=[
                        annual_yield_tasks[scenario_name],
                        crop_value_tasks[scenario_name],
                    ],
                    target_path_list=[str(annual_yield_value_raster_path)],
                )

                scenario_variables["annual_yield_value_raster_path"][
                    scenario_name
                ] = annual_yield_value_raster_path
            else:
                # Calculate delta yield
                for season in scenario_variables["season_list"]:
                    LOGGER.info(
                        f"{scenario_name} | Calculating Seasonal Crop Yield Marginal Value raster: {season}"
                    )
                    delta_seasonal_yield_raster_path = season_dir / (
                        _DELTA_SEASONAL_YIELD_BY_VISITATION_FILE_PATTERN
                        % (season, file_suffix)
                    )

                    delta_seasonal_yield_tasks[scenario_name][
                        season
                    ] = task_graph.add_task(
                        task_name=f"calculate_delta_yield_{season}_{scenario_name}",
                        func=pygeoprocessing.raster_calculator,
                        args=(
                            [
                                (
                                    str(
                                        scenario_variables[
                                            "crop_pollinator_dependence_raster_path"
                                        ][scenario_name][season]
                                    ),
                                    1,
                                ),
                                (
                                    str(
                                        scenario_variables[
                                            "half_saturation_coefficient_raster_path"
                                        ][scenario_name]
                                    ),
                                    1,
                                ),
                                (
                                    str(
                                        scenario_variables[
                                            "total_pollinator_visitation_index_path"  # NOTE this is annual visitation, not seasonal
                                        ][BASELINE_SCENARIO_LABEL]
                                    ),
                                    1,
                                ),
                            ],
                            delta_yield_op,
                            str(delta_seasonal_yield_raster_path),
                            gdal.GDT_Float32,
                            _INDEX_NODATA,
                        ),
                        target_path_list=[str(delta_seasonal_yield_raster_path)],
                        dependent_task_list=[
                            crop_pollinator_dependence_tasks[scenario_name][season]
                            for season in scenario_variables["season_list"]
                        ]
                        + [half_saturation_tasks[scenario_name]],
                    )

                    scenario_variables["delta_seasonal_yield_raster_path"][
                        scenario_name
                    ][season] = delta_seasonal_yield_raster_path

                # calculate the marginal change in annual crop yield
                LOGGER.info(
                    f"{scenario_name} | Calculating Annual Crop Yield Marginal Value raster (by change in visitation)"
                )
                delta_seasonal_yield_raster_path_band_list = [
                    (
                        str(
                            scenario_variables["delta_seasonal_yield_raster_path"][
                                scenario_name
                            ][season]
                        ),
                        1,
                    )
                    for season in scenario_variables["season_list"]
                ]
                delta_annual_yield_raster_path = output_dir / (
                    _DELTA_ANNUAL_YIELD_FILE_PATTERN % file_suffix
                )

                delta_annual_yield_tasks[scenario_name] = task_graph.add_task(
                    task_name=f"calculate_delta_annual_crop_yield_{scenario_name}",
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        delta_seasonal_yield_raster_path_band_list,
                        sum_raster_op,  # QUESTION switch with sum_raster_cap_one_op
                        str(delta_annual_yield_raster_path),
                        gdal.GDT_Float32,
                        _INDEX_NODATA,
                    ),
                    target_path_list=[str(delta_annual_yield_raster_path)],
                    dependent_task_list=[
                        delta_seasonal_yield_tasks[scenario_name][season]
                        for season in scenario_variables["season_list"]
                    ],
                )

                scenario_variables["delta_annual_yield_raster_path"][
                    scenario_name
                ] = delta_annual_yield_raster_path

                LOGGER.info(
                    f"{scenario_name} | Convolving Annual Crop Yield Marginal Value raster"
                )
                delta_annual_yield_convolution_raster_path = output_dir / (
                    _DELTA_ANNUAL_YIELD_BY_LANDSCAPE_SCORE_FILE_PATTERN % file_suffix
                )

                delta_annual_yield_convolution_tasks[
                    scenario_name
                ] = task_graph.add_task(
                    task_name=f"calculate_delta_yield_by_landscape_score_{scenario_name}",
                    func=pygeoprocessing.convolve_2d,
                    args=(
                        [
                            (
                                str(
                                    scenario_variables[
                                        "delta_annual_yield_raster_path"
                                    ][scenario_name]
                                ),
                                1,
                            ),
                            (str(kernel_path), 1),
                            str(delta_annual_yield_convolution_raster_path),
                        ]
                    ),
                    kwargs={
                        "ignore_nodata_and_edges": True,
                        "mask_nodata": True,
                        "normalize_kernel": False,
                    },
                    target_path_list=[str(delta_annual_yield_convolution_raster_path)],
                    dependent_task_list=[delta_annual_yield_tasks[scenario_name]],
                )

                scenario_variables["delta_annual_yield_convolution_raster_path"][
                    scenario_name
                ] = delta_annual_yield_convolution_raster_path

                LOGGER.info(
                    f"{scenario_name} | Calculating Annual Crop Yield Marginal Value raster (by change in habitat)"
                )

                delta_annual_yield_habitat_raster_path = output_dir / (
                    _DELTA_ANNUAL_YIELD_BY_HABITAT_FILE_PATTERN % file_suffix
                )

                delta_annual_yield_habitat_tasks[scenario_name] = task_graph.add_task(
                    task_name=f"calculate_delta_yield_by_habitat_{scenario_name}",
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        [
                            (
                                str(
                                    scenario_variables[
                                        "delta_annual_yield_convolution_raster_path"
                                    ][scenario_name]
                                ),
                                1,
                            ),
                            (
                                str(
                                    scenario_variables[
                                        "d_pollinator_landscape_score_path"
                                    ][scenario_name]
                                ),
                                1,
                            ),
                            (
                                str(
                                    scenario_variables[
                                        "annual_yield_value_raster_path"
                                    ][BASELINE_SCENARIO_LABEL]
                                ),
                                1,
                            ),
                        ],
                        delta_yield_habitat_op,
                        str(delta_annual_yield_habitat_raster_path),
                        gdal.GDT_Float32,
                        _INDEX_NODATA,
                    ),
                    target_path_list=[str(delta_annual_yield_habitat_raster_path)],
                    dependent_task_list=[
                        delta_seasonal_yield_tasks[scenario_name][season]
                        for season in scenario_variables["season_list"]
                    ],
                )

                scenario_variables["delta_annual_yield_habitat_raster_path"][
                    scenario_name
                ] = delta_annual_yield_habitat_raster_path

                # calculate the marginal change in annual crop yield value
                LOGGER.info(
                    f"{scenario_name} | Calculating Annual Crop Yield Value Marginal Value raster"
                )
                delta_annual_yield_value_raster_path = output_dir / (
                    _DELTA_ANNUAL_YIELD_VALUE_FILE_PATTERN % file_suffix
                )

                delta_annual_yield_value_tasks[scenario_name] = task_graph.add_task(
                    task_name=f"calculate_delta_annual_yield_value_{scenario_name}",
                    func=pygeoprocessing.raster_calculator,
                    args=(
                        [
                            (
                                str(
                                    scenario_variables[
                                        "delta_annual_yield_habitat_raster_path"
                                    ][scenario_name]
                                ),
                                1,
                            ),
                            (
                                str(
                                    scenario_variables["crop_value_raster_path"][
                                        scenario_name
                                    ]
                                ),
                                1,
                            ),
                        ],
                        multiply_rasters_op,
                        str(delta_annual_yield_value_raster_path),
                        gdal.GDT_Float32,
                        _INDEX_NODATA,
                    ),
                    dependent_task_list=[
                        delta_annual_yield_tasks[scenario_name],
                        crop_value_tasks[scenario_name],
                    ],
                    target_path_list=[str(delta_annual_yield_value_raster_path)],
                )

                scenario_variables["delta_annual_yield_value_raster_path"][
                    scenario_name
                ] = delta_annual_yield_value_raster_path

    task_graph.close()
    task_graph.join()


# Create raster multiplication operations from nodata value
multiply_rasters_op = rasterops.MultiplyRasters(_INDEX_NODATA)
sum_raster_op = rasterops.SumRasters(_INDEX_NODATA)
sum_raster_cap_one_op = rasterops.SumRastersWithCap(1, _INDEX_NODATA)


def d_pollinator_landscape_score_task_op(
    kernel_path,
    species_abundance,
    baseline_local_nesting_density_path,
    baseline_foraging_effectiveness_path,
    baseline_seasonal_forage_path,
    scenario_seasonal_forage_path,
    baseline_habitat_nesting_suitability_path,
    scenario_habitat_nesting_suitability_path,
    marginal_value_path,
):
    """wrapper function for marginal value raster calculation, to embed kernel array analysis inside a TaskGraph task"""
    kernel_array = pygeoprocessing.raster_to_numpy_array(str(kernel_path))
    mid_kernel_weight = np.max(kernel_array)
    marginal_value_op = _DeltaPollinatorLandscapeScoreOp(species_abundance)
    pygeoprocessing.raster_calculator(
        [
            (str(baseline_local_nesting_density_path), 1),
            (str(baseline_foraging_effectiveness_path), 1),
            (str(baseline_seasonal_forage_path), 1),
            (str(scenario_seasonal_forage_path), 1),
            (str(baseline_habitat_nesting_suitability_path), 1),
            (str(scenario_habitat_nesting_suitability_path), 1),
            (mid_kernel_weight, "raw"),
        ],
        marginal_value_op,
        str(marginal_value_path),
        gdal.GDT_Float32,
        _INDEX_NODATA,
    )


class _DeltaPollinatorLandscapeScoreOp(object):
    """Calculate dpls/dphab =
       d_forage * nesting_density + d_nest * forage + (d_nest + d_forage) * center of convolution window.
    # TODO flesh out these docstrings."""

    def __init__(self, species_abundance):
        """Create a closure for species abundance to multiply later.

        Args:
            species_abundance (float): value to use in `__call__` when
                calculating pollinator abundance.

        Returns:
            None.
        """
        self.species_abundance = species_abundance
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(_DeltaPollinatorLandscapeScoreOp.__call__).encode(
                    "utf-8"
                )
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = _DeltaPollinatorLandscapeScoreOp.__name__
        self.__name__ += str(species_abundance)

    def __call__(
        self,
        baseline_local_nesting_density_array,
        baseline_foraging_effectiveness_array,
        baseline_seasonal_forage_array,
        scenario_seasonal_forage_array,
        baseline_habitat_nesting_suitability_array,
        scenario_habitat_nesting_suitability_array,
        mid_kernel_weight,
    ):
        """Calculate [
            (d_forage * nesting_density) +
            (d_nesting * forage) +
            (d_nesting + d_forage) * mid_kernel_weight
        ] * self.species_abundance.

        where:
            d_forage = change in annual foraging effectiveness (scenario - baseline)
            nesting_density = nesting density within foraging radius
            d_nesting = change in habitat nesting (scenario - baseline)
            forage = annual foraging effectiveness

        dpls/dphab = (d_forage * nest_scape) + (d_nest * forage) + (d_nest + d_forage) *center of convolution window

        """
        result = np.empty_like(baseline_foraging_effectiveness_array)
        result[:] = _INDEX_NODATA
        valid_mask = baseline_foraging_effectiveness_array != _INDEX_NODATA

        d_forage = (
            scenario_seasonal_forage_array[valid_mask]
            - baseline_seasonal_forage_array[valid_mask]
        )
        nesting_density = baseline_local_nesting_density_array[valid_mask]
        d_nesting = (
            scenario_habitat_nesting_suitability_array[valid_mask]
            - baseline_habitat_nesting_suitability_array[valid_mask]
        )
        forage = baseline_foraging_effectiveness_array[valid_mask]

        result[valid_mask] = self.species_abundance * (
            (d_forage * nesting_density)
            + (d_nesting * forage)
            + (d_nesting + d_forage) * mid_kernel_weight
        )

        return result


class _CalculateHabitatNestingIndex(object):
    """Closure for HN(x, s) = max_n(N(x, n) ns(s,n)) calculation."""

    def __init__(
        self,
        substrate_path_map,
        species_substrate_index_map,
        target_habitat_nesting_index_path,
    ):
        """Define parameters necessary for HN(x,s) calculation.

        Args:
            substrate_path_map (dict): map substrate name to substrate index
                raster path. (N(x, n))
            species_substrate_index_map (dict): map substrate name to
                scalar value of species substrate suitability. (ns(s,n))
            target_habitat_nesting_index_path (string): path to target
                raster
        """
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(_CalculateHabitatNestingIndex.__call__).encode(
                    "utf-8"
                )
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = _CalculateHabitatNestingIndex.__name__
        self.__name__ += str(
            [
                substrate_path_map,
                species_substrate_index_map,
                target_habitat_nesting_index_path,
            ]
        )
        self.substrate_path_list = [
            str(substrate_path_map[substrate_id])
            for substrate_id in sorted(substrate_path_map)
        ]

        self.species_substrate_suitability_index_array = np.array(
            [
                species_substrate_index_map[substrate_id]
                for substrate_id in sorted(substrate_path_map)
            ]
        ).reshape((len(species_substrate_index_map), 1))

        self.target_habitat_nesting_index_path = target_habitat_nesting_index_path

    def __call__(self):
        """Calculate HN(x, s) = max_n(N(x, n) ns(s,n))."""

        def max_op(*substrate_index_arrays):
            """Return the max of index_array[n] * ns[n]."""
            result = np.max(
                np.stack([x.flatten() for x in substrate_index_arrays])
                * self.species_substrate_suitability_index_array,
                axis=0,
            )
            result = result.reshape(substrate_index_arrays[0].shape)
            nodata_mask = utils.array_equals_nodata(
                substrate_index_arrays[0], _INDEX_NODATA
            )
            result[nodata_mask] = _INDEX_NODATA
            return result

        pygeoprocessing.raster_calculator(
            [(str(x), 1) for x in self.substrate_path_list],
            max_op,
            str(self.target_habitat_nesting_index_path),
            gdal.GDT_Float32,
            _INDEX_NODATA,
        )


def delta_yield_op(
    scenario_crop_pollinator_dependence_array,
    scenario_half_saturation_coefficient_array,
    baseline_pollinator_visitation_array,
):
    """Calculate dYield_dPV = k * cd * (1+k) / ((v+k) * (v+k))
    where:
        cd = crop dependence on pollinators
        k = half saturation coefficient
        v = Pollinator Visitation
    """
    # QUESTION Update this to respond to changes in (i.e. delta_) crop pollinator dependence, half saturation, and crop value?
    # QUESTION Currently it is only responsive to changes in pollinator visitation.
    # QUESTION Basically, what happens if you change the crop type as well as the adjacent habitat?

    result = np.empty_like(scenario_crop_pollinator_dependence_array)
    result[:] = _INDEX_NODATA
    denominator = result.copy()

    valid_mask = np.logical_and.reduce(
        (
            scenario_crop_pollinator_dependence_array != _INDEX_NODATA,
            scenario_half_saturation_coefficient_array != _INDEX_NODATA,
            baseline_pollinator_visitation_array != _INDEX_NODATA,
        )
    )

    k = scenario_half_saturation_coefficient_array[valid_mask]
    v = baseline_pollinator_visitation_array[valid_mask]

    # Check denominator for zeros and remove from valid mask
    denominator[valid_mask] = (v + k) * (v + k)
    valid_mask &= denominator != 0

    cd = scenario_crop_pollinator_dependence_array[valid_mask]
    k = scenario_half_saturation_coefficient_array[valid_mask]
    v = baseline_pollinator_visitation_array[valid_mask]
    d = denominator[valid_mask]

    result[valid_mask] = (k * cd * (1 + k)) / d
    return result


def yield_op(
    crop_pollinator_dependence_array,
    half_saturation_coefficient_array,
    pollinator_visitation_array,
):
    """Calculate yield = (1 - cd) + (1 + k) * cd * PV / (k + PV)
    where:
        cd = crop dependence on pollinators
        k = half saturation coefficient
        PV = Pollinator Visitation

    """

    result = np.empty_like(crop_pollinator_dependence_array)
    result[:] = _INDEX_NODATA
    denominator = result.copy()

    valid_mask = np.logical_and.reduce(
        (
            crop_pollinator_dependence_array != _INDEX_NODATA,
            half_saturation_coefficient_array != _INDEX_NODATA,
            pollinator_visitation_array != _INDEX_NODATA,
        )
    )

    k = half_saturation_coefficient_array[valid_mask]
    v = pollinator_visitation_array[valid_mask]

    # Check denominator for zeros and remove from valid mask
    denominator[valid_mask] = k + v
    valid_mask &= denominator != 0

    cd = crop_pollinator_dependence_array[valid_mask]
    k = half_saturation_coefficient_array[valid_mask]
    v = pollinator_visitation_array[valid_mask]
    d = denominator[valid_mask]

    result[valid_mask] = (1 - cd) + (1 + k) * cd * v / d

    # Original code from Eric:
    # crop_yield = (
    #     (100 - crop_pollinator_dependence_array)
    #     + (100 + 10)
    #     * (crop_pollinator_dependence_array)
    #     * (pollinator_visitation_array / 100)
    #     / (10 + pollinator_visitation_array)
    # ) / 100

    return result


def delta_yield_habitat_op(
    delta_yield_by_delta_landscape_score_array,
    delta_landscape_score_by_habitat_array,
    yield_array,
):
    """Calculate change in yields by habitat = dy_dpls * dpls_dhab * y
    where:
        dy_dpls = delta yield over delta landscape score
        dpls_dhab = delta landscape score over delta habitat
        y = baseline crop yield
    """
    # QUESTION: Why is there an old version multiplied by Yield? I believe we decided to not do this but good to note.

    result = np.empty_like(delta_yield_by_delta_landscape_score_array)
    result[:] = _INDEX_NODATA

    valid_mask = np.logical_and.reduce(
        (
            delta_yield_by_delta_landscape_score_array != _INDEX_NODATA,
            delta_landscape_score_by_habitat_array != _INDEX_NODATA,
            # yield_array != _INDEX_NODATA,
        )
    )

    dy_dpls = delta_yield_by_delta_landscape_score_array[valid_mask]
    dpls_dhab = delta_landscape_score_by_habitat_array[valid_mask]
    # y = yield_array[valid_mask]

    # result[valid_mask] = dy_dpls * dpls_dhab * y
    result[valid_mask] = dy_dpls * dpls_dhab

    return result


def _parse_scenario_variables(
    args, scenario_labels_list, scenario_args_dict, scenario_filepath_list
):
    """Parse out scenario variables from input parameters.

    This function parses through the guild table, biophysical table, and
    farm polygons (if available) to generate

    Parameter:
        args (dict): this is the args dictionary passed in to the `execute`
            function, requires a 'guild_table_path' key.
        scenario_labels_list (list of string): list of scenario labels
        scenario_args_dict (dict): dictionary of scenario arguments
        scenario_filepath_list (list of string): list of scenario filepaths

    Returns:
        A dictionary with the keys:
            * season_list (list of string)
            * substrate_list (list of string)
            * species_list (list of string)
            * alpha_value[species] (float)
            * landcover_substrate_index[substrate][landcover] (float)
            * landcover_floral_resources[season][landcover] (float)
            * species_abundance[species] (string->float)
            * species_foraging_activity[(species, season)] (string->float)
            * species_substrate_index[(species, substrate)] (tuple->float)
            * foraging_activity_index[(species, season)] (tuple->float)
    """

    yield_calc_bool = "calculate_yield" in args and args["calculate_yield"]

    # Update MODEL_SPEC if necessary
    if yield_calc_bool:
        MODEL_SPEC["args"]["landcover_biophysical_table_path"]["columns"].update(
            {
                "crop_pollinator_dependence_[SEASON]_index": {
                    "type": "ratio",
                    "about": "Index of crop dependence on pollinators during [SEASON]. "
                    "Replace [SEASON] with season names matching those "
                    "in the Guild Table, so that there is a column for each "
                    "season.",
                },
                "half_saturation_coefficient": {
                    "type": "ratio",
                    "about": "The half saturation coefficient for the crop grown "
                    "in this area. This is the wild pollinator abundance "
                    "(i.e. the proportion of all pollinators that are "
                    "wild) needed to reach half of the total potential "
                    "pollinator-dependent yield.",
                },
                "value_per_hectare": {
                    "type": "number",
                    "units": u.currency / u.hectare,
                    "about": "The value of the crop in dollars per square hectare. "
                    "This is used to calculate the value of the pollinator-"
                    "dependent yield.",
                },
            },
        )

    guild_table_path = args["guild_table_path"]

    guild_df = utils.read_csv_to_dataframe(
        guild_table_path, MODEL_SPEC["args"]["guild_table_path"]
    )

    LOGGER.info("Checking to make sure guild table has all expected headers")
    for header in _EXPECTED_GUILD_HEADERS:
        matches = re.findall(header, " ".join(guild_df.columns))
        if len(matches) == 0:
            raise ValueError(
                "Expected a header in guild table that matched the pattern "
                f"'{header}' but was unable to find one. Here are all the "
                f"headers from {guild_table_path}: {guild_df.columns}"
            )

    # this dict to dict will map seasons to guild/biophysical headers
    # ex season_to_header['spring']['guilds']
    season_to_header = collections.defaultdict(dict)
    # this dict to dict will map substrate types to guild/biophysical headers
    # ex substrate_to_header['cavity']['biophysical']
    substrate_to_header = collections.defaultdict(dict)
    for header in guild_df.columns:
        match = re.match(_FORAGING_ACTIVITY_RE_PATTERN, header)
        if match:
            season = match.group(1)
            season_to_header[season]["guild"] = match.group()
        match = re.match(_NESTING_SUITABILITY_PATTERN, header)
        if match:
            substrate = match.group(1)
            substrate_to_header[substrate]["guild"] = match.group()

    # Create empty dicts for scenario names
    substrate_to_header[substrate]["biophysical"] = []
    season_to_header[season]["biophysical"] = []
    season_to_header[season]["crop_pollinator_dependence"] = []
    # Iterate through scenarios
    scenario_biophysical_df_dict = {}
    for scenario_name in scenario_labels_list:
        landcover_biophysical_df = utils.read_csv_to_dataframe(
            scenario_args_dict["landcover_biophysical_table_path"][scenario_name],
            MODEL_SPEC["args"]["landcover_biophysical_table_path"],
        )

        scenario_biophysical_df_dict[scenario_name] = landcover_biophysical_df

        LOGGER.info(
            f"{scenario_name} | Checking to make sure biophysical table has all expected headers"
        )
        biophysical_table_headers = landcover_biophysical_df.columns

        bio_headers = (
            _EXPECTED_BIOPHYSICAL_YIELD_HEADERS
            if yield_calc_bool
            else _EXPECTED_BIOPHYSICAL_HEADERS
        )
        for header in bio_headers:
            matches = re.findall(header, " ".join(biophysical_table_headers))
            if len(matches) == 0:
                raise ValueError(
                    "Expected a header in biophysical table that matched the "
                    f"pattern '{header}' but was unable to find one. Here are all "
                    f"the headers from {scenario_args_dict['landcover_biophysical_table_path'][scenario_name]}: "
                    f"{', '.join(biophysical_table_headers)}"
                )

        for header in biophysical_table_headers:
            match = re.match(_FLORAL_RESOURCES_AVAILABLE_RE_PATTERN, header)
            if match:
                season = match.group(1)
                season_to_header[season]["biophysical"].append(match.group())
            match = re.match(_NESTING_SUBSTRATE_RE_PATTERN, header)
            if match:
                substrate = match.group(1)
                substrate_to_header[substrate]["biophysical"].append(match.group())
            if yield_calc_bool:
                match = re.match(_CROP_POLLINATOR_DEPENDENCE_RE_PATTERN, header)
                if match:
                    season = match.group(1)
                    season_to_header[season]["crop_pollinator_dependence"].append(
                        match.group()
                    )

    season_to_header[season]["biophysical"] = list(
        set(season_to_header[season]["biophysical"])
    )
    substrate_to_header[substrate]["biophysical"] = list(
        set(substrate_to_header[substrate]["biophysical"])
    )
    if yield_calc_bool:
        season_to_header[season]["crop_pollinator_dependence"] = list(
            set(season_to_header[season]["crop_pollinator_dependence"])
        )

    header_check = [
        len(season_to_header[season]["biophysical"]) == 1,
        len(substrate_to_header[substrate]["biophysical"]) == 1,
    ]
    assert_message = (
        f"Expected a single column name for all tables. {substrate_to_header[substrate]['biophysical']}, "
        f"{season_to_header[season]['biophysical']} "
    )
    if yield_calc_bool:
        header_check.append(
            len(season_to_header[season]["crop_pollinator_dependence"]) == 1
        )
        assert_message += f"{season_to_header[season]['crop_pollinator_dependence']}"

    assert all(header_check), assert_message

    season_to_header[season]["biophysical"] = season_to_header[season]["biophysical"][0]
    substrate_to_header[substrate]["biophysical"] = substrate_to_header[substrate][
        "biophysical"
    ][0]
    if yield_calc_bool:
        season_to_header[season]["crop_pollinator_dependence"] = season_to_header[
            season
        ]["crop_pollinator_dependence"][0]

    # Check season_to_header for completeness
    for table_type, lookup_table in season_to_header.items():
        if yield_calc_bool and len(lookup_table) != 3:
            raise ValueError(
                "Expected a two biophysical entries (floral, crop pollinator dependence) and a "
                f"guild entry for '{table_type}' but instead found only {list(lookup_table.keys())}. "
                f"Ensure there are corresponding entries of '{table_type}' in "
                "both the guilds table and the biophysical table (for floral and crop pollinator dependence)."
            )
        elif len(lookup_table) != 2 and (
            "calculate_yield" not in args or not args["calculate_yield"]
        ):
            raise ValueError(
                f"Expected a biophysical and guild entry for '{table_type}' "
                f"but instead found only {list(lookup_table.keys())}. Ensure there are "
                f"corresponding entries of '{table_type}' in both the guilds "
                "and biophysical table."
            )
    # Check substrate_to_header for completeness
    for table_type, lookup_table in substrate_to_header.items():
        if len(lookup_table) != 2:
            raise ValueError(
                f"Expected a biophysical, and guild entry for '{table_type}' "
                f"but instead found only {list(lookup_table.keys())}. Ensure there are "
                f"corresponding entries of '{table_type}' in both the guilds "
                "and biophysical table."
            )

    result = {}
    # * season_list (list of string)
    result["season_list"] = sorted(season_to_header)
    # * substrate_list (list of string)
    result["substrate_list"] = sorted(substrate_to_header)
    # * species_list (list of string)
    result["species_list"] = sorted(guild_df.index)

    result["alpha_value"] = dict()
    for species in result["species_list"]:
        result["alpha_value"][species] = guild_df[_ALPHA_HEADER][species]

    # * species_abundance[species] (string->float)
    total_relative_abundance = guild_df[_RELATIVE_SPECIES_ABUNDANCE_FIELD].sum()
    result["species_abundance"] = {}
    for species in result["species_list"]:
        result["species_abundance"][species] = (
            guild_df[_RELATIVE_SPECIES_ABUNDANCE_FIELD][species]
            / total_relative_abundance
        )

    # map the relative foraging activity of a species during a certain season
    # (species, season)
    result["species_foraging_activity"] = dict()
    for species in result["species_list"]:
        total_activity = np.sum(
            [
                guild_df[_FORAGING_ACTIVITY_PATTERN % season][species]
                for season in result["season_list"]
            ]
        )
        for season in result["season_list"]:
            result["species_foraging_activity"][(species, season)] = (
                guild_df[_FORAGING_ACTIVITY_PATTERN % season][species] / total_activity
            )

    # * landcover_substrate_index[substrate][landcover] (float)
    result["landcover_substrate_index"] = {
        scenario_name: {substrate: {} for substrate in result["substrate_list"]}
        for scenario_name in scenario_labels_list
    }
    for scenario_name in scenario_labels_list:
        for landcover_id, row in scenario_biophysical_df_dict[scenario_name].iterrows():
            for substrate in result["substrate_list"]:
                substrate_biophysical_header = substrate_to_header[substrate][
                    "biophysical"
                ]
                result["landcover_substrate_index"][scenario_name][substrate][
                    landcover_id
                ] = row[substrate_biophysical_header]

    # * landcover_floral_resources[season][landcover] (float)
    result["landcover_floral_resources"] = {
        scenario_name: {season: {} for season in result["season_list"]}
        for scenario_name in scenario_labels_list
    }
    for scenario_name in scenario_labels_list:
        for landcover_id, row in scenario_biophysical_df_dict[scenario_name].iterrows():
            for season in result["season_list"]:
                floral_resources_header = season_to_header[season]["biophysical"]
                result["landcover_floral_resources"][scenario_name][season][
                    landcover_id
                ] = row[floral_resources_header]

    # * species_substrate_index[(species, substrate)] (tuple->float)
    result["species_substrate_index"] = {
        species: {substrate: {} for substrate in result["substrate_list"]}
        for species in result["species_list"]
    }
    for species in result["species_list"]:
        for substrate in result["substrate_list"]:
            substrate_guild_header = substrate_to_header[substrate]["guild"]
            result["species_substrate_index"][species][substrate] = guild_df[
                substrate_guild_header
            ][species]

    # * foraging_activity_index[(species, season)] (tuple->float)
    result["foraging_activity_index"] = {}
    for species in result["species_list"]:
        for season in result["season_list"]:
            key = (species, season)
            foraging_biophysical_header = season_to_header[season]["guild"]
            result["foraging_activity_index"][key] = guild_df[
                foraging_biophysical_header
            ][species]

    if yield_calc_bool:
        # * crop_pollinator_dependence[season][landcover] (float)
        result["crop_pollinator_dependence"] = {
            scenario_name: {season: {} for season in result["season_list"]}
            for scenario_name in scenario_labels_list
        }
        for scenario_name in scenario_labels_list:
            for landcover_id, row in scenario_biophysical_df_dict[
                scenario_name
            ].iterrows():
                for season in result["season_list"]:
                    crop_pollinator_dependence_header = season_to_header[season][
                        "crop_pollinator_dependence"
                    ]
                    result["crop_pollinator_dependence"][scenario_name][season][
                        landcover_id
                    ] = row[crop_pollinator_dependence_header]

        # * half_saturation_coefficient[landcover] (float)
        result["half_saturation_coefficient"] = {
            scenario_name: {} for scenario_name in scenario_labels_list
        }
        for scenario_name in scenario_labels_list:
            for landcover_id, row in scenario_biophysical_df_dict[
                scenario_name
            ].iterrows():
                result["half_saturation_coefficient"][scenario_name][
                    landcover_id
                ] = row[_HALF_SATURATION_FIELD]

        # * crop_value[landcover] (float)
        result["crop_value"] = {
            scenario_name: {} for scenario_name in scenario_labels_list
        }
        for scenario_name in scenario_labels_list:
            for landcover_id, row in scenario_biophysical_df_dict[
                scenario_name
            ].iterrows():
                result["crop_value"][scenario_name][landcover_id] = row[
                    _CROP_VALUE_FIELD
                ]

    # Create filepath dictionaries
    for scenario_filepath in scenario_filepath_list:
        result[scenario_filepath] = {name: {} for name in scenario_labels_list}

    return result