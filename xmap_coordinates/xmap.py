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

        self.interp_dims = tuple()

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
        for k in self._obj.dims:
            if k in coords.keys():
                if not isinstance(coords[k], xr.DataArray):
                    try:
                        cleaned[k] = da_atleast1d(coords[k], k)
                    except ValueError:
                        raise ValueError("ndarrays must have ndim==1")
                else:
                    # Strip breadcrumb coordinates
                    cleaned[k] = coords[k].reset_coords(drop=True)

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
        shape_interp = coords[initial].shape

        # Find non-interpolation dimensions and coordinates
        other_dims = tuple(set(self._obj.dims).difference(coords.keys()))
        other_coords = {k: self._obj.coords[k] for k in other_dims}

        # Since interpolation vectors (i.e. coords) can have different top-level dimensions than self._obj, we need
        # to use the dimensions from coords for the interpolation dimensions. And, due to coords being meshgrids, we
        # need to use the underlying coordinates.
        self.output_dims = other_dims + tuple(coords[initial].coords.keys())
        self.output_coords = {**other_coords, **coords[initial].coords}

        # It is also handy to keep track of the same coordinates and dims but in reference to the data itself. This
        # will allow us to transpose the data prior to interpolation
        self.interp_dims = other_dims + tuple(coords.keys())

        output_shape = tuple(self._obj.shape[self._obj.dims.index(k)] for k in other_dims) + shape_interp
        da_output = xr.DataArray(np.zeros(output_shape), coords=self.output_coords, dims=self.output_dims)
        return da_output

    def interp(
        self, output: Optional[xr.DataArray] = None, kwargs_map: Optional[dict] = None, **coords
    ) -> xr.DataArray:

        default_map = dict(order=3, mode="nearest", prefilter=True)
        kwargs_map = {} if kwargs_map is None else kwargs_map
        kwargs_map = {**default_map, **kwargs_map}

        return output

    def _coords2pixels(self, **coords) -> Dict[str, xr.DataArray]:
        pass


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

    # TODO: What to do about coordinates that are unsorted (both in the da and in the interpolation dimensions)
    default_map = dict(order=3, mode="nearest", prefilter=True)
    if kwargs_map is not None:
        kwargs_map = {**default_map, **kwargs_map}
    else:
        kwargs_map = default_map

    # Verify that all interpolation coordinates are present in the data to be interpolated
    if not set(coords_kwargs.keys()).issubset(set(da.dims)):
        raise ValueError(
            "One or more interpolation coordinate is not a dimensions in the DataArray to be interpolated:\n"
            f"Data Array dims: {list(da.dims)}\n"
            f"Interpolation dims: {list(coords_kwargs.keys())}"
        )

    # TODO: Verify that all coordinates in `da` are only 1-D. `da` must be on a regular grid for any interpolation to
    # make sense

    # TODO: Enforce increasing coordinates
    # np.all(np.diff(xp) > 0)

    # The partial shape of `da` (ignoring the uninterpolated dims) and the corresponding vectors that
    # describe the regular grid that `da` will be interpolated to
    shape_result = []  # if da_output is None else da_output.shape
    coords_result = {}  # if da_output is None else {k: v for k, v in zip(da_output.dims, da_output.coords.values())}
    keys_result = []

    # coords_xarray mapped to the pixel grid
    coords_pixel = {}

    # Meshgrid pixel coords
    coords_meshed = {}

    # TODO: If da_output is provided ensure it's coordinates line up with the underlying coordinates in coords_kwargs
    if da_output is not None:
        for ii, (k, v) in enumerate(coords_kwargs.items()):
            # They must be xarrays
            if not isinstance(v, xr.DataArray):
                raise ValueError()
            # The coordinates must be a subset of the output's dims
            if not set(v.coords.keys()).issubset(da_output.dims):
                raise ValueError("Impromperly formatted coordinates between interpolation coords and da_output.")
            # The underlying coordinates must be the same!
            if ii == 0:
                keys_result = list(v.coords.dims)
                shape_result = v.shape
                coords_result = dict(v.coords)

                # These must line up with the output shape's coords
                for kk, vv in coords_result.items():
                    assert np.all(da_output[kk] == vv)
            else:
                if not dict_equal(coords_result, dict(v.coords)):
                    raise ValueError("Coords do not match!")

    else:
        # Convert all coordinates to xarrays and ascertain the shape of the resulting interpolated data
        for k in da.dims:
            if k in coords_kwargs.keys():
                if not isinstance(coords_kwargs[k], xr.DataArray):
                    coord = np.atleast_1d(coords_kwargs[k])
                    if coord.ndim != 1:
                        raise ValueError("ndarrays must have a dimension equal to 1")
                    coords_kwargs[k] = xr.DataArray(coord, coords={k: coord}, dims=(k,))
                else:
                    # Strip breadcrumb coordinates
                    coords_kwargs[k] = coords_kwargs[k].reset_coords(drop=True)

                idx = coords_kwargs[k].dims.index(k)
                shape_result.append(coords_kwargs[k].shape[idx])
                coords_result[k] = coords_kwargs[k][k]

    # Build the output array
    if da_output is None:
        other_dims = list(set(da.dims).difference(coords_kwargs.keys()))
        other_coods = {da.coords[k] for k in other_dims}
        end_shape = [da.shape[da.dims.index(k)] for k in other_dims] + shape_result
        end_coords = {**other_coods, **coords_result}

        da_output = xr.DataArray(np.zeros(end_shape), coords=end_coords, dims=other_dims + coords_result.keys())

    # Scale interpolation vectors to coordinates
    # At this point, all coordinates are xarrays, so we can interpolate using xarray's interp function
    coords_pixel = {}
    for k in da.dims:
        if k in coords_kwargs:
            # Dimensions where the underlying unitary coordinates are a special case. They do not need to be
            # interpolated.
            if da.coords[k].shape == (1,):
                coords_pixel[k] = xr.DataArray(
                    np.zeros(coords_kwargs[k].shape), coords=coords_kwargs[k].coords, dims=(list(coords_kwargs[k].dims))
                )
            else:
                # coordinates may be N-D. Because of this, the coordinates must be stacked, interpolated to the pixel
                # map and then unstacked.
                # TODO: Stack only if coordinates are ND
                coord = np.ravel(coords_kwargs[k])
                f_interp = interp1d(
                    # da coordinates may be N-D, so we have to grab the 1D vector that describes the N-D coordinate
                    np.atleast_1d(da.coords[k]),
                    np.arange(0, len(da.coords[k]), 1),
                    kind="linear",
                    fill_value="extrapolate",
                )
                coords_pixel[k] = xr.DataArray(
                    np.reshape(f_interp(coord), coords_kwargs[k].shape),
                    coords=coords_kwargs[k].coords,
                    dims=list(coords_kwargs[k].dims),
                )

    # For any coordinate that is 1D, broadcast that coordinate to the N-D shape
    # TODO: Is there a better way of doing this?
    dummy = xr.DataArray(np.zeros(shape_result), coords=coords_result, dims=keys_result)
    for k in da.dims:
        if k in coords_pixel:
            # This forces dimension order as well, so a transpose / reshape is unnecessary here.
            coords_meshed[k] = coords_pixel[k].broadcast_like(dummy)

    # TODO: idx_coords appears to be unused. Why?
    # Flatten coordinates
    # idx_coords = {}
    interp_shape = dummy.shape
    mg_coords = [x.values.flatten() for x in list(coords_meshed.values())]
    pts = np.array(list(zip(*mg_coords))).T

    # Reshape dataarray so that we can loop over the dimensions that will not be interpolated
    looped_dims = list(set(da.dims).difference(coords_meshed.keys()))

    # TODO: Stacking seems to set a lower bound here. Look into replacing with numpy reshapes and see if things get
    # faster
    if looped_dims:
        da_stacked = da.stack(looped=looped_dims)
    else:
        # Support case where all of da's dimensions are present in coords_kwargs.
        da_stacked = da.expand_dims("looped", axis=-1)

    da_result = da_output.stack(looped=looped_dims)
    for ii in range(da_stacked.coords["looped"].size):
        da_result.values[..., ii] = map_coordinates(
            da_stacked[..., ii].values, pts, output=da.dtype, **kwargs_map
        ).reshape(interp_shape)

    if looped_dims:
        da_result = da_result.unstack().transpose(*da_output.dims)
    else:
        da_result = da_result.isel(looped=0, drop=True).transpose(*da_output.dims)

    # da_result = da_result.assign_coords(**coords_result)
    return da_result
