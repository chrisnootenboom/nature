import logging
import typing

from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
from natcap.invest import utils
import pygeoprocessing

from ... import nature
from .. import param_utils


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

pathlike = str | Path

NLCD_LUCODES = {
    "Open Water": 11,
    "Perennial Ice/Snow": 12,
    "Developed, Open Space": 21,
    "Developed, Low Intensity": 22,
    "Developed, Medium Intensity": 23,
    "Developed, High Intensity": 24,
    "Barren Land": 31,
    "Deciduous Forest": 41,
    "Evergreen Forest": 42,
    "Mixed Forest": 43,
    "Shrub/Scrub": 52,
    "Grassland/Herbaceous": 71,
    "Pasture/Hay": 81,
    "Cultivated Crops": 82,
    "Woody Wetlands": 90,
    "Emergent Herbaceous Wetlands": 95,
}

nlud_to_management_csv = Path(__file__).parent / "nlud_management.csv"
nlud_to_management_csv = Path(__file__).parent / "nlud_management.csv"
nlud_to_management_csv = Path(__file__).parent / "nlud_management.csv"

# TODO public_access: for relevant NLCD values, for applicable NLUD categories, replace public access with value from NLUD_management_df

# TODO green area: for relevant NLCD values, for applicable NLUD categories, replace green area with value from NLUD_management_df

# TODO irrigation: for relevant NLCD values, for applicable NLUD categories, replace irrigation with value from NLUD_management_df
