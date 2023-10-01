from pathlib import Path


N_WORKERS = -1

# InVEST models
CARBON_ARGS = {
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
}
POLLINATION_ARGS = {
    r"workspace_dir": "",
    r"results_suffix": "",
    r"n_workers": N_WORKERS,
    r"landcover_raster_path": "",
    r"guild_table_path": "",
    r"landcover_biophysical_table_path": "",
    r"farm_vector_path": r"",
}
URBAN_COOLING_ARGS = {
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
}
URBAN_FLOOD_RISK_ARGS = (
    {
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
)

# Misc models
WHO5_MENTAL_HEALTH_ARGS = {
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
}

# Marginal Value models
POLLINATION_MV_ARGS = {
    "workspace_dir": "",
    "results_suffix": "",
    "landcover_raster_path": "",
    "guild_table_path": "",
    "landcover_biophysical_table_path": "",
    "scenario_labels_list": [],
    "scenario_landcover_biophysical_table_path_list": [],
    "scenario_landcover_raster_path_list": [],
    "calculate_yield": False,
    # "farm_vector_path": None,
    "aggregate_size": None,
    "n_workers": N_WORKERS,
}


# Valuation modules
URBAN_COOLING_VALUATION_ARGS = {
    "workspace_dir": "",
    "results_suffix": "",
    "city": "",
    "lulc_tif": "",
    "air_temp_tif": "",
    "dd_energy_path": "",
    "mortality_risk_path": "",
}
