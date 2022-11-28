import unittest

import numpy as np
import xarray as xr

import xmap_coordinates  # noqa: F401
from xmap_coordinates.utils import da_atleast1d


class TestCreateOutput(unittest.TestCase):
    def setUp(self) -> None:
        self.da = xr.DataArray(
            np.zeros((3, 2, 2)), coords={"x": np.arange(3), "y": np.arange(2), "z": np.arange(2)}, dims=("x", "y", "z")
        )

        return super().setUp()

    def test_normal(self):
        x = np.arange(10)
        y = np.arange(6)
        result = self.da.xmap._cleanse(x=x, y=y)
        da = self.da.xmap._create_output(**result)
        print(da.dims)
        # self.assertTupleEqual(da.dims, ('z', 'x', 'y'))

    def test_different_coords(self):
        a = da_atleast1d(np.arange(10), "a")
        b = da_atleast1d(np.arange(6), "b")

        m_coords = xr.broadcast(a, b)
        coords = {"x": m_coords[0], "y": m_coords[1]}

        result = self.da.xmap._cleanse(**coords)
        da = self.da.xmap._create_output(**result)
        print(da.dims)


if __name__ == "__main__":
    unittest.main()
