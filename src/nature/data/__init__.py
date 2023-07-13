import pkgutil
import io

import pandas as pd

# This file will read in the data files from the data directory as VARIABLES for import


def _read_csv(filename):
    return pd.read_csv(io.BytesIO(pkgutil.get_data(__name__, filename)))


BIVARIATE_COLORMAP = _read_csv("/colormaps/bivariate.csv")
