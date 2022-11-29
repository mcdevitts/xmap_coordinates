from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from .utils import da_atleast1d, dict_equal

__all__ = ("xmap_coordinates", "XmapCoordinates")

"""
Current implementation only allows N-D arrays if an output DataArray is provided. Otherwise, the coords must be 1D.
How do we want to handle this going forward?

"""


@xr.register_dataarray_accessor("xmap")
class XmapCoordinates:
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj
        self._center = None

        self.output_dims = tuple()
        self.output_coords = {}

        #
        self.interp_dims = tuple()

        # Information for extra (non-interpolating) dimensions
        self.other_dims = tuple()
        self.other_coords = {}

        # dims and shape of interpolation coordinates arranged in an order that matches self._obj
        self.coords_dims = tuple()
        self.coords_shape = tuple()

    def _cleanse(self, **coords: Dict[str, Union[xr.DataArray, npt.ArrayLike]]) -> Dict[str, xr.DataArray]:
        """Cast incoming coords to multi-dimensional xarrays.

        Given N `coords`, N N-dimensional xarrays will be created with the coords and dims corresponding to the keys
        and values of `coords`. This method will also clean up passed ndarrays or xarrays:
        - Cast to meshgrided DataArrays
        - Reorder coords to match order of `self._obj`
        - Reorder the underlying coordinates of coords (sorry) to match `self._obj`
        - Strip unnecessary information from coords DataArrays
        """

        cleaned = {}
        for k in self.coords_dims:
            if not isinstance(coords[k], xr.DataArray):
                try:
                    cleaned[k] = da_atleast1d(coords[k], k)
                except ValueError:
                    raise ValueError("ndarrays must have ndim==1")
            else:
                # Strip breadcrumb coordinates
                cleaned[k] = coords[k].transpose(*self.coords_dims).reset_coords(drop=True)

        # Ensure all entries have the same shape
        initial = list(cleaned.values())[0]
        assert all([v.ndim == initial.ndim for v in list(cleaned.values())])
        # TODO: Ensure all N-D arrays have the same coordinates

        # Broadcast arrays if arrays are not already meshgrids
        if initial.ndim == 1:
            m_cleaned = xr.broadcast(*cleaned.values())
            for ii, k in enumerate(cleaned):
                cleaned[k] = m_cleaned[ii]

        # TODO: Do we need xarrays or could we live with ndarrays?
        return cleaned

    def _create_output(self, **coords: Dict[str, xr.DataArray]) -> xr.DataArray:
        """Create output DataArray given the interpolation coordinates `coords`.

        The DataArrays within `coords` are meshgrids of the interpolation variables.
        """
        initial = list(coords.keys())[0]
        self.coords_shape = coords[initial].shape

        # Since interpolation vectors (i.e. coords) can have different top-level dimensions than self._obj, we need
        # to use the dimensions from coords for the interpolation dimensions. And, due to coords being meshgrids, we
        # need to use the underlying coordinates.
        self.output_dims = self.other_dims + tuple(coords[initial].coords.keys())
        self.output_coords = {**self.other_coords, **coords[initial].coords}

        # It is also handy to keep track of the same coordinates and dims but in reference to the data itself. This
        # will allow us to transpose the data prior to interpolation
        self.interp_dims = self.other_dims + tuple(coords.keys())

        output_shape = tuple(self._obj.shape[self._obj.dims.index(k)] for k in self.other_dims) + self.coords_shape
        da_output = xr.DataArray(np.zeros(output_shape), coords=self.output_coords, dims=self.output_dims)
        return da_output

    @staticmethod
    def _pixelate(x: xr.DataArray, y: xr.DataArray, xarray: bool = False) -> Union[xr.DataArray, np.ndarray]:
        """ """
        # Dimensions where the underlying unitary coordinates are a special case. They do not need to be
        # interpolated.
        if x.shape == (1,):
            pixels = np.zeros(y.shape)
        else:
            # coordinates are n-dimensional. Because of this, the coordinates must be stacked, interpolated to
            # the pixel map and then unstacked.
            coord = np.ravel(y)
            f_interp = interp1d(
                # da coordinates may be N-D, so we have to grab the 1D vector that describes the N-D coordinate
                np.atleast_1d(x),
                np.arange(0, len(x), 1),
                kind="linear",
                fill_value="extrapolate",
            )
            pixels = np.reshape(f_interp(coord), y.shape)
        if xarray:
            pixels = xr.DataArray(pixels, coords=y.coords, dims=(list(y.dims)))
        return pixels

    def interp(
        self, output: Optional[xr.DataArray] = None, kwargs_map: Optional[dict] = None, **coords
    ) -> xr.DataArray:

        # Default arguments for map_coordinates
        default_map = dict(order=3, mode="nearest", prefilter=True)
        kwargs_map = {} if kwargs_map is None else kwargs_map
        kwargs_map = {**default_map, **kwargs_map}

        # TODO: Is there a better way to organize the dimensions, coordinates, and shape of the various groupings?
        # Find interpolation dimensions in the order they appear in the data
        self.coords_dims = tuple(x for x in self._obj.dims if x in coords.keys())

        # Find non-interpolation dimensions and coordinates in the order they appear in the data
        self.other_dims = tuple(x for x in self._obj.dims if x not in coords.keys())
        self.other_coords = {k: self._obj.coords[k] for k in self.other_dims}

        # Clean up the coordinates, and create meshgrids if they aren't already
        coords = self._cleanse(**coords)

        # Infer the output array from the input coordinates
        da_output = self._create_output(**coords)

        # map_coordinates works on pixel grid coordinates. Translate coordinates to pixel coordinates before
        # interpolating
        pixels = {k: self._pixelate(self._obj.coords[k], coords[k], xarray=False) for k in self.coords_dims}
        mg_pixels = [x.flatten() for x in list(pixels.values())]
        pts = np.array(list(zip(*mg_pixels))).T

        # TODO: Reshape self._obj so that it's dimensions are: [*self.other_dims, *self.coords_dims]
        # TODO: Reshape da_output as well

        # TODO: Force c-contiguous order for speed? Does it help?

        # TODO: Stacking seems to set a lower bound here. Look into replacing with numpy reshapes and see if things get
        # faster
        # TODO: I think this can be replaced with a simple reshape now.
        if self.other_dims:
            da_stacked = self._obj.stack(looped=self.other_dims)
        else:
            # Support case where all of da's dimensions are present in coords_kwargs.
            da_stacked = self._obj.expand_dims("looped", axis=-1)

        da_result = da_output.stack(looped=self.other_dims)
        for ii in range(da_stacked.coords["looped"].size):
            da_result.values[..., ii] = map_coordinates(
                da_stacked[..., ii].values, pts, output=self._obj.dtype, **kwargs_map
            ).reshape(self.coords_shape)

        if self.other_dims:
            da_result = da_result.unstack().transpose(*da_output.dims)
        else:
            da_result = da_result.isel(looped=0, drop=True).transpose(*da_output.dims)

        # TODO: Reshape da_output back to shape that is coherent with the order of self._obj

        return da_result


def xmap_coordinates(
    da: xr.DataArray, da_output: Optional[xr.DataArray] = None, kwargs_map=None, **coords_kwargs
) -> xr.DataArray:
    """Generic interpolater for DataArrays that uses scipy's map_coordinates.

    This function supports the following:
        - Higher-order interpolation for N-D arrays
        - Definable extrapolation methods for N-D arrays
        - Smart support for unitary dimensions (i.e. it doesn't barf)
        - Support for additional, uninterpolated dimensions
        - Real or complex data

    The data to be interpolated must be regularly gridded - the underlying coordinates must all have a single dimension.
    The requested coordinates (`coords_kwargs`) may be regularly gridded or irregularly grided.

    Parameters
    ----------
    da : xr.DataArray
        Data to be interpolated. Underlying coordinates for `da` must be one dimensional (i.e. regularly gridded).
        `da` may be either real or complex, and it may have unitary dimensions as well as extra dimensions.
    da_output : xr.DataArray, optional

    kwargs_map : dict, opt
        Settings passed directly to map_coordinates. The default settings are:
            order=3
            mode="nearest"
            prefilter=True
    coords_kwargs : dict of ndarray or xr.DataArray
        Each coordinate may either be 1-D or N-D. However if it is N-D, the shape of the other dimensions must match
        the shape of the other coordinates in `coords_kwargs`.

    Returns
    -------
    da_result : xr.DataArray
        Resulting shape will be the `da`'s shape with the specified dimensions matching the lengths of the coordinates
        in `coords_kwargs`.
        dtype will be identical to `da`

    """

    pass
