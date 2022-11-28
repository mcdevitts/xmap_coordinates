import unittest

import numpy as np
import xarray as xr

import xmap_coordinates  # noqa: F401
from xmap_coordinates.utils import da_atleast1d


class TestCleanse(unittest.TestCase):
    def setUp(self) -> None:
        self.da = xr.DataArray(
            np.zeros((3, 2, 2)), coords={"x": np.arange(3), "y": np.arange(2), "z": np.arange(2)}, dims=("x", "y", "z")
        )

        return super().setUp()

    def check(self, da, **kwargs):

        keys = tuple(kwargs.keys())

        for ii, var in enumerate(keys):
            # Check dims and order
            self.assertTupleEqual(da[var].dims, keys)
            # self.assertDictEqual(dict(result[var].coords), {"x": x, "y": y})
            m_grid = np.meshgrid(*kwargs.values(), indexing="xy")
            np.testing.assert_array_almost_equal(da[var].data, m_grid[ii].T)

    def test_cleanse_ndarray_1d(self):
        x = np.arange(10)
        y = np.arange(6)
        result = self.da.xmap._cleanse(x=x, y=y)
        self.check(result, x=x, y=y)

        # Order doesn't matter
        result = self.da.xmap._cleanse(y=y, x=x)
        self.check(result, x=x, y=y)

    def test_cleanse_ndarray_nd(self):
        x = np.arange(10)
        y = np.arange(6)
        m_xy = np.meshgrid(x, y)
        with self.assertRaises(ValueError):
            self.da.xmap._cleanse(x=m_xy[0], y=m_xy[1])

    def test_cleanse_xarray_1d(self):
        x = da_atleast1d(np.arange(10), "x")
        y = da_atleast1d(np.arange(6), "y")

        result = self.da.xmap._cleanse(x=x, y=y)
        self.check(result, x=x, y=y)

    def test_cleanse_xarray_nd(self):
        x = da_atleast1d(np.arange(10), "x")
        y = da_atleast1d(np.arange(6), "y")

        m_coords = xr.broadcast(x, y)
        coords = {"x": m_coords[0], "y": m_coords[1]}

        result = self.da.xmap._cleanse(**coords)
        self.check(result, x=x, y=y)


if __name__ == "__main__":
    unittest.main()
