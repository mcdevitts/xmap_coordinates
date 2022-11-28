import unittest

import numpy as np
import xarray as xr

import xmap_coordinates


def da_atleast1d(x, name) -> xr.DataArray:
    """Convert a float or 1D ndarray to an xarray whose coordinates are the same values as the data."""
    if isinstance(x, xr.DataArray):
        return x
    else:
        x = np.atleast_1d(x)
        return xr.DataArray(x, coords={name: x}, dims=(name,))


class TestValidate(unittest.TestCase):
    def setUp(self) -> None:
        self.da = xr.DataArray(
            np.zeros((3, 2, 2)), coords={"x": np.arange(3), "y": np.arange(2), "z": np.arange(2)}, dims=("x", "y", "z")
        )

        return super().setUp()

    def test_cleanse_ndarray_1d(self):
        pass

    def test_cleanse_ndarray_nd(self):
        pass

    def test_cleanse_xarray_1d(self):
        x = da_atleast1d(np.arange(10), "x")
        y = da_atleast1d(np.arange(6), "y")
        # m_xy = xr.broadcast(x, y)
        # d_xy = dict(x=m_xy[0], y=m_xy[1])
        # self.da.xmap._validate(x=x, y=y)
        self.da.xmap._cleanse(x=x, y=y)

    def test_cleanse_xarray_nd(self):
        pass


if __name__ == "__main__":
    unittest.main()
