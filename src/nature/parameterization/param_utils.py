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


MODEL_PARAMETERS = {
    "urban_cooling_model": [
        "shade",
        "kc",
        "albedo",
        "green_area",
        "building_intensity",
    ],
    "urban_carbon": [
        "c_above",
        "c_below",
        "c_soil",
        "c_dead",
        "c_embedded",
        "c_emissions",
    ],
    "pollination": [
        "nesting_all_availability_index",
        "floral_resources_annual_index",
    ],
    "urban_nature_access": ["urban_nature", "search_radius_m"],
    "ndr": [
        "load_n",
        "load_p",
        "eff_n",
        "eff_p",
        "crit_len_n",
        "crit_len_p",
        "proportion_subsurface_n",
        "proportion_subsurface_p",
    ],
}


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


class ThresholdReplace(object):
    """Class wrapper around the 'replace' operator: replaces current parameter value with new,
    if another column exceeds a threshold.
    Allows application to different management practices inputs for the same method"""

    def __init__(self, output_column, threshold_column, threshold):
        self.output_column = output_column
        self.threshold_column = threshold_column
        self.threshold = threshold

    def __call__(self, row, parameter, nodata_value):
        """returns a new parameter value where the management practice is applicable
        if the threshold is exceeded"""
        return (
            row[self.output_column]
            if row[self.output_column] != nodata_value
            and row[self.threshold_column] >= self.threshold
            else row[parameter]
        )


class WeightedAverage(object):
    """Class wrapper around the 'weighted_average' operator: takes the weighted average of two values.
    Allows application to different management practices inputs for the same method"""

    def __init__(self, reference_value, percent: float | str):
        self.reference_value = reference_value
        self.percent = percent

    def __call__(self, row, parameter, nodata_value):
        """returns a weighted average"""
        if isinstance(self.percent, float):
            output = (
                row[parameter] * (1 - self.percent)
                + self.reference_value * self.percent
            )
        else:
            output = (
                row[parameter] * (1 - row[self.percent])
                + self.reference_value * row[self.percent]
            )
        return output


class PercentChange(object):
    """Class wrapper around the 'percent_change' operator: adjust a parameter by a
    specified percent (e.g., an increase of 10%: 1.1, 40% of the original value: 0.4, etc.)
    Allows application to different management practices inputs for the same method"""

    def __init__(self, percent):
        self.percent = percent

    def __call__(self, row, parameter, nodata_value):
        """returns a weighted addition to the baseline"""
        return row[parameter] * self.percent


class VariablePercentChange(object):
    """Class wrapper around the 'percent_change' operator: adjust a parameter by a maximum of a
    specified percent (e.g., an increase of 10%: 1.1, 40% of the original value: 0.4, etc.)
    but how far along to that maximum is determined by a column in the dataframe, for any number of
    maximums and columns.

    Weights must be specified as columns in the input dataframe (e.g., for adding a maximum of 10
    based on the column "fertilizer", where the column "fertilizer" is a 0 to 1 index
    of how much of 10 to add).

    Allows application to different management practices inputs for the same method"""

    def __init__(self, **kwargs):
        """kwargs are in the format of {column: percentage}"""
        for attr in kwargs.keys():
            self.__dict__[attr] = kwargs[attr]

    def __call__(self, row, parameter, nodata_value):
        """returns a multi-percentage change to the baseline. Assumes no interaction between variables"""
        for attr in self.__dict__.keys():
            if attr not in row.keys():
                raise KeyError(f"Specified column '{attr}' is not in the dataframe.")

        output = row[parameter]
        for attr, val in self.__dict__.items():
            if row[attr] != nodata_value:
                output *= 1 - (1 - val) * row[attr]

        return output


class Additive(object):
    """Class wrapper around the 'addition' operator: adds a weighted amount
    of a new value to the existing, for any number of new values. Weights must
    be specified as columns in the input dataframe (e.g., for adding a maximum of 10
    based on the column "fertilizer", where the column "fertilizer" is a 0 to 1 index
    of how much of 10 to add).
    Allows application to different management practices inputs for the same method"""

    def __init__(self, **kwargs):
        """kwargs are in the format of {column: weight}"""
        for attr in kwargs.keys():
            self.__dict__[attr] = kwargs[attr]

    def __call__(self, row, parameter, nodata_value):
        """returns a multi-weighted addition to the baseline. Assumes no interaction between variables"""
        for attr in self.__dict__.keys():
            if attr not in row.keys():
                raise KeyError(f"Specified column '{attr}' is not in the dataframe.")

        output = row[parameter]
        for attr, val in self.__dict__.items():
            if row[attr] != nodata_value:
                output += row[attr] * val

        return output


class Replace(object):
    """Class wrapper around the 'replace' operator: replaces current parameter value with new.
    Allows application to different management practices inputs for the same method"""

    def __init__(self, output_column):
        self.output_column = output_column

    def __call__(self, row, parameter, nodata_value):
        """returns a new parameter value where the management practice is applicable"""
        return (
            row[self.output_column]
            if row[self.output_column] != nodata_value
            else row[parameter]
        )
