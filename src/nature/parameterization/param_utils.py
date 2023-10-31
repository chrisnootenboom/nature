import logging
import typing

from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
from natcap.invest import utils
import pygeoprocessing

from .. import nature


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

pathlike = str | Path

# BMP Labels

HEDGEROW_LABEL = "Hedgerow"
FLORAL_STRIP_LABEL = "Floral Strip"
BUFFER_STRIP_LABEL = "Buffer Strip / Vegetated Filter Strip"
PRAIRIE_RESTORATION_LABEL = "Prairie Restoration"
MAX_PARAMETER_LABEL = "Maximum Parameters"
GRASSED_WATERWAY_LABEL = "Grassed Waterway"
COVER_CROP_LABEL = "Cover Crop"
SEASONAL_COVER_CROP_LABEL = "Seasonal Cover Crop"
STRIPS_LABEL = "STRIPS"


def _replicate_parameter_row(parameter_row_df: pd.DataFrame, lucode: int, label: str):
    """Creates a pandas df row for 'Buffer Strips' that mimics the input parameters
    Args:
        parameter_row_df (Pandas DataFrame): Single-row dataframe to replicate for the BMP
        lucode (int): LULC code to assign to the BMP
        label (str): Name of the BMP

    Returns:
        Pandas DataFrame: Row with parameter values for BMP
    """
    if len(parameter_row_df) != 1:
        logger.warning(
            "Multiple rows provided for parameterization. Selecting only the first."
        )
    result = parameter_row_df.head(1).copy()
    result["lucode"], result["lulc_name"] = (
        lucode,
        label,
    )
    return result
