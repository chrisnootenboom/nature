import logging
from pathlib import Path
import typing
from typing import List, Set, Dict, Tuple, Optional
import tempfile
import shutil
import pickle
import warnings
import yaml

import requests
import xmltodict
from bs4 import BeautifulSoup

import math
import random
import pandas as pd
import numpy as np
import pint

import matplotlib.colors

import rasterio as rio
from rasterio.merge import merge
from rasterio.mask import mask
from osgeo import gdal, osr
import geopandas as gpd
from osgeo import ogr
from rasterstats import zonal_stats

import pygeoprocessing
import pygeoprocessing.kernels
import natcap.invest.utils
import natcap.invest.spec_utils
import natcap.invest.validation

import nature
import nature.zonal_statistics
import nature.equity
import nature.functions

from nature.models import monthly_ucm, ucm_valuation

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

pathlike = str | Path


# Import config
config = "/Users/cnootenboom/repos/nature/tests/configs/tests.yaml"
with open(config) as yaml_data_file:
    args = yaml.load(yaml_data_file, Loader=yaml.FullLoader)

workspace_path = Path(args["workspace_dir"])

# Population density
# nature.equity.census_2020_population_density(
#     args["census block dir 2020"],
#     workspace_path,
#     args["template raster"],
#     state_fips_list=args["state fips list"],
#     per_pixel=True,
#     mosaic=False,
# )

# Monthly UCM
# monthly_ucm.execute(args)

# Zonal stats
# nature.zonal_statistics.zonal_stats(
#     "/Users/cnootenboom/repos/nature/tests/nature/workspace/cdd_test_Jun.tif",
#     args["building_vector_path"],
#     join_to_vector=True,
#     output_vector_path=workspace_path / "zonal_stats.gpkg",
# )


# Equity index
# equity_label = "equity"
# es_label = "es"
# es_column = "x"
# equity_columns = ["a", "b", "c"]
# df = pd.DataFrame(
#     {
#         "x": [1, 2, 3, 4, 5, 6],
#         "a": [10, 10, 0, 0, 0, 10],
#         "b": [10, 10, 10, 10, 10, 10],
#         "c": [10, 10, 10, 10, 10, 10],
#     }
# )

# equity_df, equity_score = nature.equity.equity_covariance(
#     df, equity_columns, es_column, equity_label, es_label
# )

# City-wide equity assessment
nature.equity.area_equity_assessment(
    args["census_table_labels"],
    args["census_table_path_list"],
    args["es_vector_path"],
    args["es_field_labels"],
    args["es_fields"],
    args["census_join_field"],
    args["workspace_path"],
)
