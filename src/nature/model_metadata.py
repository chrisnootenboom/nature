import dataclasses

N_WORKERS = -1


import natcap.invest.carbon


@dataclasses.dataclass
class _MODELMETA_NCI:
    """Dataclass to store frequently used model metadata."""

    model_title: str  # display name for the model
    pyname: str  # importable python module name for the model
    aliases: tuple  # alternate names for the model, if any
    args: dict  # arguments for the model


@dataclasses.dataclass
class _MODELMETA:
    """Dataclass to store frequently used model metadata."""

    model_title: str  # display name for the model
    pyname: str  # importable python module name for the model
    gui: str  # importable python class for the corresponding Qt UI
    userguide: str  # name of the corresponding built userguide file
    aliases: tuple  # alternate names for the model, if any


NCI_METADATA = {
    "pollination_mv": _MODELMETA_NCI(
        model_title="Pollination Marginal Value",
        pyname="nature.mv.nci_pollination",
        aliases=(),
        args={
            "workspace_dir": "",
            "results_suffix": "",
            "landcover_raster_path": "",
            "guild_table_path": "",
            "landcover_biophysical_table_path": "",
            "scenario_labels_list": [],
            "scenario_landcover_biophysical_table_path_list": [],
            "scenario_landcover_raster_path_list": [],
            "calculate_yield": False,
            "farm_vector_path": None,
            "aggregate_size": None,
            "n_workers": N_WORKERS,
        },
    ),
    "ndr_mv": _MODELMETA_NCI(
        model_title="Nutrient Delivery Ratio Marginal Value",
        pyname="nature.mv.nci_ndr",
        aliases=(),
        args={},
    ),
    "urban_cooling_mv": _MODELMETA_NCI(
        model_title="Urban Cooling Marginal Value",
        pyname="nature.mv.nci_urban_cooling",
        aliases=(),
        args={},
    ),
    "urban_cooling_valuation": _MODELMETA_NCI(
        model_title="Urban Cooling Valuation",
        pyname="nature.models.ucm_valuation",
        aliases=(),
        args={
            "workspace_dir": "",
            "results_suffix": "",
            "city": "",
            "lulc_raster_path": "",
            "air_temp_tif": "",
            "building_vector_path": "",
            "building_energy_table_path": "",
            "mortality_risk_path": "",
        },
    ),
    "monthly_ucm": _MODELMETA_NCI(
        model_title="Monthly Urban Cooling",
        pyname="nature.models.monthly_ucm",
        aliases=(),
        args={
            r"workspace_dir": "",
            r"results_suffix": "",
            r"n_workers": N_WORKERS,
            r"lulc_raster_path": "",
            r"ref_eto_raster_path": "",
            r"aoi_vector_path": "",
            r"biophysical_table_path": "",
            r"green_area_cooling_distance": "",
            r"t_air_average_radius": "",
            r"uhi_max": "",
            r"do_energy_valuation": False,
            r"do_productivity_valuation": False,
            r"building_vector_path": r"",
            r"cc_method": "",
            r"cc_weight_shade": "",
            r"cc_weight_albedo": "",
            r"cc_weight_eti": "",
            "air_temp_dir": "",
            "year": "",
            "city": "",
            "building_energy_table_path": "",
            "mortality_risk_path": "",
        },
    ),
}

MISC_MODEL_METADATA = {
    "who5_mental_health": _MODELMETA_NCI(
        model_title="WHO5 Mental Health",
        pyname="nature.models.who5_mental_health",
        aliases=(),
        args={
            "workspace_dir": "",
            "results_suffix": "",
            "n_workers": N_WORKERS,
            "population_raster_path": "",
            "ndvi_baseline_raster_path": "",
            "ndvi_scenario_raster_paths": "",
            "scenario_suffixes": "",
            "impact_distance_m": "",
            "mean_who5": "",
            "baseline_ndvi": "",
            "ndvi_increment": "",
            "per_capita_expenditures": "",
        },
    ),
}

INVEST_METADATA = {
    "carbon": _MODELMETA_NCI(
        model_title="Carbon Storage and Sequestration",
        pyname="natcap.invest.carbon",
        aliases=(),
        args={
            r"workspace_dir": "",
            r"results_suffix": "",
            r"n_workers": N_WORKERS,
            r"lulc_cur_path": "",
            r"calc_sequestration": False,
            r"lulc_fut_path": r"",
            r"do_redd": False,
            r"lulc_redd_path": r"",
            r"carbon_pools_path": "",
            r"lulc_cur_year": r"",
            r"lulc_fut_year": r"",
            r"do_valuation": False,
            r"price_per_metric_ton_of_c": r"",
            r"discount_rate": r"",
            r"rate_change": r"",
        },
    ),
    "pollination": _MODELMETA_NCI(
        model_title="Crop Pollination",
        pyname="natcap.invest.pollination",
        aliases=(),
        args={
            r"workspace_dir": "",
            r"results_suffix": "",
            r"n_workers": N_WORKERS,
            r"landcover_raster_path": "",
            r"guild_table_path": "",
            r"landcover_biophysical_table_path": "",
            r"farm_vector_path": r"",
        },
    ),
    "urban_flood_risk_mitigation": _MODELMETA_NCI(
        model_title="Urban Flood Risk Mitigation",
        pyname="natcap.invest.urban_flood_risk_mitigation",
        aliases=("ufrm",),
        args={
            r"workspace_dir": r"",
            r"results_suffix": "",
            r"n_workers": N_WORKERS,
            r"aoi_watersheds_path": "",
            r"rainfall_depth": "",
            r"lulc_path": "",
            r"soils_hydrological_group_raster_path": "",
            r"curve_number_table_path": "",
            r"built_infrastructure_vector_path": r"",
            r"infrastructure_damage_loss_table_path": r"",
        },
    ),
    "urban_cooling_model": _MODELMETA_NCI(
        model_title="Urban Cooling",
        pyname="natcap.invest.urban_cooling_model",
        aliases=("ucm",),
        args={
            r"workspace_dir": "",
            r"results_suffix": "",
            r"n_workers": N_WORKERS,
            r"lulc_raster_path": "",
            r"ref_eto_raster_path": "",
            r"aoi_vector_path": "",
            r"biophysical_table_path": "",
            r"green_area_cooling_distance": "",
            r"t_air_average_radius": "",
            r"t_ref": "",
            r"uhi_max": "",
            r"do_energy_valuation": False,
            r"do_productivity_valuation": False,
            r"avg_rel_humidity": r"",
            r"building_vector_path": r"",
            r"energy_consumption_table_path": r"",
            r"cc_method": "",
            r"cc_weight_shade": "",
            r"cc_weight_albedo": "",
            r"cc_weight_eti": "",
        },
    ),
}

_UNPROCESSED_INVEST_METADATA = {
    "annual_water_yield": _MODELMETA(
        model_title="Annual Water Yield",
        pyname="natcap.invest.annual_water_yield",
        gui="annual_water_yield.AnnualWaterYield",
        userguide="annual_water_yield.html",
        aliases=("hwy", "awy"),
    ),
    "coastal_blue_carbon": _MODELMETA(
        model_title="Coastal Blue Carbon",
        pyname="natcap.invest.coastal_blue_carbon.coastal_blue_carbon",
        gui="cbc.CoastalBlueCarbon",
        userguide="coastal_blue_carbon.html",
        aliases=("cbc",),
    ),
    "coastal_blue_carbon_preprocessor": _MODELMETA(
        model_title="Coastal Blue Carbon Preprocessor",
        pyname="natcap.invest.coastal_blue_carbon.preprocessor",
        gui="cbc.CoastalBlueCarbonPreprocessor",
        userguide="coastal_blue_carbon.html",
        aliases=("cbc_pre",),
    ),
    "coastal_vulnerability": _MODELMETA(
        model_title="Coastal Vulnerability",
        pyname="natcap.invest.coastal_vulnerability",
        gui="coastal_vulnerability.CoastalVulnerability",
        userguide="coastal_vulnerability.html",
        aliases=("cv",),
    ),
    "crop_production_percentile": _MODELMETA(
        model_title="Crop Production: Percentile",
        pyname="natcap.invest.crop_production_percentile",
        gui="crop_production.CropProductionPercentile",
        userguide="crop_production.html",
        aliases=("cpp",),
    ),
    "crop_production_regression": _MODELMETA(
        model_title="Crop Production: Regression",
        pyname="natcap.invest.crop_production_regression",
        gui="crop_production.CropProductionRegression",
        userguide="crop_production.html",
        aliases=("cpr",),
    ),
    "delineateit": _MODELMETA(
        model_title="DelineateIt",
        pyname="natcap.invest.delineateit.delineateit",
        gui="delineateit.Delineateit",
        userguide="delineateit.html",
        aliases=(),
    ),
    "forest_carbon_edge_effect": _MODELMETA(
        model_title="Forest Carbon Edge Effect",
        pyname="natcap.invest.forest_carbon_edge_effect",
        gui="forest_carbon.ForestCarbonEdgeEffect",
        userguide="carbon_edge.html",
        aliases=("fc",),
    ),
    "habitat_quality": _MODELMETA(
        model_title="Habitat Quality",
        pyname="natcap.invest.habitat_quality",
        gui="habitat_quality.HabitatQuality",
        userguide="habitat_quality.html",
        aliases=("hq",),
    ),
    "habitat_risk_assessment": _MODELMETA(
        model_title="Habitat Risk Assessment",
        pyname="natcap.invest.hra",
        gui="hra.HabitatRiskAssessment",
        userguide="habitat_risk_assessment.html",
        aliases=("hra",),
    ),
    "ndr": _MODELMETA(
        model_title="Nutrient Delivery Ratio",
        pyname="natcap.invest.ndr.ndr",
        gui="ndr.Nutrient",
        userguide="ndr.html",
        aliases=(),
    ),
    "recreation": _MODELMETA(
        model_title="Visitation: Recreation and Tourism",
        pyname="natcap.invest.recreation.recmodel_client",
        gui="recreation.Recreation",
        userguide="recreation.html",
        aliases=(),
    ),
    "routedem": _MODELMETA(
        model_title="RouteDEM",
        pyname="natcap.invest.routedem",
        gui="routedem.RouteDEM",
        userguide="routedem.html",
        aliases=(),
    ),
    "scenario_generator_proximity": _MODELMETA(
        model_title="Scenario Generator: Proximity Based",
        pyname="natcap.invest.scenario_gen_proximity",
        gui="scenario_gen.ScenarioGenProximity",
        userguide="scenario_gen_proximity.html",
        aliases=("sgp",),
    ),
    "scenic_quality": _MODELMETA(
        model_title="Scenic Quality",
        pyname="natcap.invest.scenic_quality.scenic_quality",
        gui="scenic_quality.ScenicQuality",
        userguide="scenic_quality.html",
        aliases=("sq",),
    ),
    "sdr": _MODELMETA(
        model_title="Sediment Delivery Ratio",
        pyname="natcap.invest.sdr.sdr",
        gui="sdr.SDR",
        userguide="sdr.html",
        aliases=(),
    ),
    "seasonal_water_yield": _MODELMETA(
        model_title="Seasonal Water Yield",
        pyname="natcap.invest.seasonal_water_yield.seasonal_water_yield",
        gui="seasonal_water_yield.SeasonalWaterYield",
        userguide="seasonal_water_yield.html",
        aliases=("swy",),
    ),
    "stormwater": _MODELMETA(
        model_title="Urban Stormwater Retention",
        pyname="natcap.invest.stormwater",
        gui="stormwater.Stormwater",
        userguide="stormwater.html",
        aliases=(),
    ),
    "wave_energy": _MODELMETA(
        model_title="Wave Energy Production",
        pyname="natcap.invest.wave_energy",
        gui="wave_energy.WaveEnergy",
        userguide="wave_energy.html",
        aliases=(),
    ),
    "wind_energy": _MODELMETA(
        model_title="Wind Energy Production",
        pyname="natcap.invest.wind_energy",
        gui="wind_energy.WindEnergy",
        userguide="wind_energy.html",
        aliases=(),
    ),
    "urban_nature_access": _MODELMETA(
        model_title="Urban Nature Access",
        pyname="natcap.invest.urban_nature_access",
        gui="urban_nature_access.UrbanNatureAccess",
        userguide="urban_nature_access.html",
        aliases=("una",),
    ),
}

NCI_MODELS = NCI_METADATA | MISC_MODEL_METADATA | INVEST_METADATA
