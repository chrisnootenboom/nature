from pathlib import Path
import typing
import hashlib
import inspect

import numpy as np

import pygeoprocessing
import natcap.invest.utils


class MultiplyRasters(object):
    """Multiply all rasters, removing nodata."""

    def __init__(self, nodata_val):
        self.nodata = nodata_val
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(MultiplyRasters.__call__).encode("utf-8")
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = MultiplyRasters.__name__

    def __call__(self, *array_list):
        """Multiply array_list and account for nodata."""
        valid_mask = np.ones(array_list[0].shape, dtype=bool)
        result = np.empty_like(array_list[0])
        result[:] = 1
        for array in array_list:
            local_valid_mask = ~natcap.invest.utils.array_equals_nodata(
                array, self.nodata
            )
            result[local_valid_mask] *= array[local_valid_mask]
            valid_mask &= local_valid_mask  # Any nodata makes the result nodata
        result[~valid_mask] = self.nodata
        return result


class SumRasters(object):
    """Sum all rasters where nodata is 0 unless the entire stack is nodata."""

    def __init__(self, nodata_val):
        self.nodata = nodata_val
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(SumRasters.__call__).encode("utf-8")
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = SumRasters.__name__

    def __call__(self, *array_list):
        """Calculate sum of array_list and account for nodata."""
        valid_mask = np.zeros(array_list[0].shape, dtype=bool)
        result = np.empty_like(array_list[0])
        result[:] = 0
        for array in array_list:
            local_valid_mask = ~natcap.invest.utils.array_equals_nodata(
                array, self.nodata
            )
            result[local_valid_mask] += array[local_valid_mask]
            valid_mask |= local_valid_mask  # Only ALL nodata makes the result nodata
        result[~valid_mask] = self.nodata
        return result


class SumRastersWithCap(object):
    """Sum all rasters where nodata is 0 unless the entire stack is nodata, but with a maximum Capped value."""

    def __init__(self, cap_val, nodata_val):
        self.nodata = nodata_val
        self.cap = cap_val
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(SumRastersWithCap.__call__).encode("utf-8")
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = SumRastersWithCap.__name__

    def __call__(self, *array_list):
        """Calculate sum of array_list and account for nodata and cap."""
        valid_mask = np.zeros(array_list[0].shape, dtype=bool)
        result = np.empty_like(array_list[0])
        result[:] = 0
        for array in array_list:
            local_valid_mask = ~natcap.invest.utils.array_equals_nodata(
                array, self.nodata
            )
            result[local_valid_mask] += array[local_valid_mask]
            valid_mask |= local_valid_mask  # Only ALL nodata makes the result nodata
        result[~valid_mask] = self.nodata
        result[result > self.cap] = self.cap
        return result


class MultiplyRasterByScalar(object):
    """Calculate a raster * scalar.  Mask through nodata."""

    def __init__(self, scalar, nodata_val):
        """Create a closure for multiplying an array by a scalar.

        Args:
            scalar (float): value to use in `__call__` when multiplying by
                its parameter.

        Returns:
            None.
        """
        self.scalar = scalar
        self.nodata = nodata_val
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(MultiplyRasterByScalar.__call__).encode("utf-8")
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = MultiplyRasterByScalar.__name__
        self.__name__ += str(scalar)

    def __call__(self, array):
        """Return array * self.scalar accounting for nodata."""
        result = np.empty_like(array)
        result[:] = self.nodata
        valid_mask = ~natcap.invest.utils.array_equals_nodata(array, self.nodata)
        result[valid_mask] = array[valid_mask] * self.scalar
        return result


class SumRastersByScalar(object):
    """Sum all rasters where nodata is 0 unless the entire stack is nodata."""

    def __init__(self, scalars, nodata_val):
        self.scalars = scalars
        self.nodata = nodata_val
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(SumRasters.__call__).encode("utf-8")
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = SumRasters.__name__

    def __call__(self, *array_list):
        """Calculate sum of array_list weighted by scalar_list and account for nodata."""
        valid_mask = np.zeros(array_list[0].shape, dtype=bool)
        result = np.empty_like(array_list[0])
        result[:] = 0
        for scalar, array in zip(self.scalars, array_list):
            local_valid_mask = ~natcap.invest.utils.array_equals_nodata(
                array, self.nodata
            )
            result[local_valid_mask] += array[local_valid_mask] * scalar
            valid_mask |= local_valid_mask  # Only ALL nodata makes the result nodata
        result[~valid_mask] = self.nodata
        return result


def mask_op(base_array, mask_array, nodata):
    result = np.copy(base_array)
    result[mask_array == nodata] = nodata
    return result
