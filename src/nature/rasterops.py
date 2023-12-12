from pathlib import Path
import typing
import hashlib
import inspect
import itertools

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


class SubtractTwoRasters(object):
    """Subtract two rasters, removing nodata."""

    def __init__(self, nodata_val):
        self.nodata = nodata_val
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(SubtractTwoRasters.__call__).encode("utf-8")
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = SubtractTwoRasters.__name__

    def __call__(self, array1, array2):
        """Subtract two rasters and account for nodata."""
        valid_mask = np.ones(array1.shape, dtype=bool)
        result = np.empty_like(array1)
        result[:] = self.nodata

        valid_mask = ~natcap.invest.utils.array_equals_nodata(
            array1, self.nodata
        ) & ~natcap.invest.utils.array_equals_nodata(array2, self.nodata)

        result[valid_mask] = array1[valid_mask] - array2[valid_mask]

        return result


class CombineRasterCategories(object):
    """Combine all rasters based on their categorical data.
    Assumes the same nodata value for all raster inputs.
    """

    def __init__(self, nodata_val, *array_values_list):
        self.nodata = nodata_val
        self.array_values = array_values_list
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(CombineRasterCategories.__call__).encode("utf-8")
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = CombineRasterCategories.__name__

    def __call__(self, *array_list):
        """Create multiple categorical raster overlay with minimum raster int type."""
        combinations_dict = {
            sum([k * (10**j) for j, k in enumerate(combo)]): i + 1
            for i, combo in enumerate(itertools.product(*self.array_values))
        }

        valid_mask = np.zeros(array_list[0].shape, dtype=bool)
        temp = np.empty_like(array_list[0], dtype=np.dtype("int16"))
        temp[:] = 0
        num_digits = 0
        for i, array in enumerate(array_list):
            local_valid_mask = ~natcap.invest.utils.array_equals_nodata(
                array, self.nodata
            )
            temp[local_valid_mask] += array.astype("int")[local_valid_mask] * (
                10**num_digits
            )
            valid_mask &= local_valid_mask  # ANY nodata makes the result nodata
            num_digits += len(str(int(np.max(array[local_valid_mask]))))
        temp[~valid_mask] = self.nodata

        result = np.empty_like(temp)
        result[:] = self.nodata
        for key, value in combinations_dict.items():
            index = np.where(temp == key)
            result[index] = value

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


class MultiplyRasterByScalarList(object):
    """Calculate raster[idx] * scalar[idx] for each idx in the paired lists."""

    def __init__(self, category_list, scalar_list, base_nodata, category_nodata):
        """Create a closure for multiplying an array by a scalar.

        Args:
            scalar (float): value to use in `__call__` when multiplying by
                its parameter.

        Returns:
            None.
        """
        self.category_list = category_list
        self.scalar_list = scalar_list
        self.base_nodata = base_nodata
        self.category_nodata = category_nodata
        # try to get the source code of __call__ so task graph will recompute
        # if the function has changed
        try:
            self.__name__ = hashlib.sha1(
                inspect.getsource(MultiplyRasterByScalarList.__call__).encode("utf-8")
            ).hexdigest()
        except IOError:
            # default to the classname if it doesn't work
            self.__name__ = MultiplyRasterByScalarList.__name__

    def __call__(self, base_array, category_array):
        result = np.empty_like(base_array)
        result[:] = self.base_nodata

        # Create valid mask
        valid_mask = ~natcap.invest.utils.array_equals_nodata(
            base_array, self.base_nodata
        ) & ~natcap.invest.utils.array_equals_nodata(
            category_array, self.category_nodata
        )

        # Multiply masked raster by category and associated scalar
        for category, scalar in zip(self.category_list, self.scalar_list):
            # Mask by category
            current_mask = np.logical_and(valid_mask, (category_array == category))

            base_array_masked = base_array[current_mask]
            result[current_mask] = base_array_masked * scalar

        return result


class SumRastersByScalar(object):
    """Sum all rasters where nodata is 0 unless the entire stack is nodata."""

    def __init__(self, scalars, array_nodata, nodata_val):
        self.scalars = scalars
        self.array_nodata = array_nodata
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
        for scalar, array, array_nodata in zip(
            self.scalars, array_list, self.array_nodata
        ):
            local_valid_mask = ~natcap.invest.utils.array_equals_nodata(
                array, array_nodata
            )
            result[local_valid_mask] += array[local_valid_mask] * scalar
            valid_mask |= local_valid_mask  # Only ALL nodata makes the result nodata
        result[~valid_mask] = self.nodata
        return result


def mask_op(base_array, mask_array, nodata):
    result = np.copy(base_array)
    result[mask_array == nodata] = nodata
    return result
