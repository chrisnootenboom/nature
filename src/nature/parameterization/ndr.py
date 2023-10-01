import logging
import typing

from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
from natcap.invest import utils
import pygeoprocessing

import nature


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

pathlike = str | Path


def solar_pollinators(
    groundcover,
    lucode,
    gcr=0.5,
):
    """Creates a pandas df row for 'Solar Pollinator' that adapts the underlying land cover parameters to solar based on density

    Args:
        groundcover (Pandas DataFrame): Single-row dataframe
        lucode (int): LULC code to assign Grassed Waterway

    Returns:
        Pandas DataFrame: Row with parameter values for the Solar Pollinator field
    """

    # Creates a pandas df row for 'Solar Pollinator Plantings' that weights parameters from the 'prairie' df based on
    # literature review

    result = groundcover.head(1).copy()
    result["lucode"], result["lulc_name"] = lucode, "Solar Pollinator Plantings"
    for parameter in list(result):
        if parameter.startswith("eff_"):
            result[parameter] = result[parameter] * (
                1 + gcr * 0.2
            )  # Assumed slight increases in nutrient uptake efficiency
    return result


# TODO Add the BMPs listed in the pollination parameterization script
