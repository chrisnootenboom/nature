from pathlib import Path
import typing
import hashlib
import inspect

import numpy as np

import pygeoprocessing
import natcap.invest.utils


class MultiplyRasters(object):
    """Multiply all rasters, removing nodata."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(MultiplyRasters.__call__).encode("utf-8")
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = MultiplyRasters.__name__

    def __call__(self, nodata_val, *array_list):
        """Multiply array_list and account for nodata."""
        valid_mask = np.ones(array_list[0].shape, dtype=bool)
        result = np.empty_like(array_list[0])
        result[:] = 1
        for array in array_list:
            local_valid_mask = ~natcap.invest.utils.array_equals_nodata(
                array, nodata_val
            )
            result[local_valid_mask] *= array[local_valid_mask]
            valid_mask &= local_valid_mask  # Any nodata makes the result nodata
        result[~valid_mask] = nodata_val
        return result


class SumRasters(object):
    """Sum all rasters where nodata is 0 unless the entire stack is nodata."""

    def __init__(self):
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(SumRasters.__call__).encode("utf-8")
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = SumRasters.__name__

    def __call__(self, nodata_val, *array_list):
        """Calculate sum of array_list and account for nodata."""
        valid_mask = np.zeros(array_list[0].shape, dtype=bool)
        result = np.empty_like(array_list[0])
        result[:] = 0
        for array in array_list:
            local_valid_mask = ~natcap.invest.utils.array_equals_nodata(
                array, nodata_val
            )
            result[local_valid_mask] += array[local_valid_mask]
            valid_mask |= local_valid_mask  # Only ALL nodata makes the result nodata
        result[~valid_mask] = nodata_val
        return result
