import dataclasses


@dataclasses.dataclass
class _MODELMETA_NCI:
    """Dataclass to store frequently used model metadata."""

    model_title: str  # display name for the model
    pyname: str  # importable python module name for the model
    aliases: tuple  # alternate names for the model, if any


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
    ),
    "ndr_mv": _MODELMETA_NCI(
        model_title="Nutrient Delivery Ratio Marginal Value",
        pyname="nature.mv.nci_ndr",
        aliases=(),
    ),
    "urban_cooling_mv": _MODELMETA_NCI(
        model_title="Urban Cooling Marginal Value",
        pyname="nature.mv.nci_urban_cooling",
        aliases=(),
    ),
}

INVEST_METADATA = {
    "annual_water_yield": _MODELMETA(
        model_title="Annual Water Yield",
        pyname="natcap.invest.annual_water_yield",
        gui="annual_water_yield.AnnualWaterYield",
        userguide="annual_water_yield.html",
        aliases=("hwy", "awy"),
    ),
    "carbon": _MODELMETA(
        model_title="Carbon Storage and Sequestration",
        pyname="natcap.invest.carbon",
        gui="carbon.Carbon",
        userguide="carbonstorage.html",
        aliases=(),
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
    "pollination": _MODELMETA(
        model_title="Crop Pollination",
        pyname="natcap.invest.pollination",
        gui="pollination.Pollination",
        userguide="croppollination.html",
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
    "urban_flood_risk_mitigation": _MODELMETA(
        model_title="Urban Flood Risk Mitigation",
        pyname="natcap.invest.urban_flood_risk_mitigation",
        gui="urban_flood_risk_mitigation.UrbanFloodRiskMitigation",
        userguide="urban_flood_mitigation.html",
        aliases=("ufrm",),
    ),
    "urban_cooling_model": _MODELMETA(
        model_title="Urban Cooling",
        pyname="natcap.invest.urban_cooling_model",
        gui="urban_cooling_model.UrbanCoolingModel",
        userguide="urban_cooling_model.html",
        aliases=("ucm",),
    ),
    "urban_nature_access": _MODELMETA(
        model_title="Urban Nature Access",
        pyname="natcap.invest.urban_nature_access",
        gui="urban_nature_access.UrbanNatureAccess",
        userguide="urban_nature_access.html",
        aliases=("una",),
    ),
}
