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
        self.da_yxz = self.da.transpose("y", "x", "z")

    def test_simple(self):
        x = np.arange(10)
        y = np.arange(6)
        result = self.da.xmap.interp(x=x, y=y)

        # TODO: test the interpolated data
        self.assertTupleEqual(result.dims, self.da.dims)

    def test_order_kwargs(self):
        x = np.arange(10)
        y = np.arange(6)
        result = self.da.xmap.interp(y=y, x=x)

        # TODO: test the interpolated data
        self.assertTupleEqual(result.dims, self.da.dims)

    def test_order_meshgrid(self):
        x = da_atleast1d(np.arange(10), "x")
        y = da_atleast1d(np.arange(6), "y")

        m_coords = xr.broadcast(x, y)
        coords = {"x": m_coords[0], "y": m_coords[1]}
        result = self.da.xmap.interp(**coords)
        self.assertTupleEqual(result.dims, self.da.dims)

        result = self.da_yxz.xmap.interp(**coords)
        self.assertTupleEqual(result.dims, self.da_yxz.dims)

    def test_order_different_underlying_coords(self):
        a = da_atleast1d(np.arange(10), "a")
        b = da_atleast1d(np.arange(6), "b")

        m_coords = xr.broadcast(a, b)
        coords = {"x": m_coords[0], "y": m_coords[1]}
        result = self.da.xmap.interp(**coords)
        self.assertTupleEqual(result.dims, ("a", "b", "z"))


if __name__ == "__main__":
    unittest.main()
