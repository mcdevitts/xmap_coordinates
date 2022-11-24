from typing import Optional

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from .utils import dict_equal

__all__ = ("xmap_coordinates",)


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
