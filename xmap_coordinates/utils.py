import numpy as np
import xarray as xr


def dict_equal(a: dict, b: dict) -> bool:
    """
    Compare two dictionaries that may contain ndarray's for equality.

    Parameters
    ----------
    a : dict
        Input dictionary
    b : dict
        Input dictionary

    Returns
    -------
    equal : bool
    """

    equal = True

    # Check if all keys match: note this is order specific
    if a.keys() != b.keys():
        equal = False
    # Check values if keys match
    else:
        # Check all instance variables
        for k in a.keys():
            # Equality checks on numpy arrays
            if isinstance(a[k], (np.ndarray, xr.DataArray)) or isinstance(b[k], (np.ndarray, xr.DataArray)):
                try:
                    equal = all(np.asarray(a[k]) == np.asarray(b[k]))
                except KeyError:
                    equal = False
            else:
                try:
                    if a[k] != b[k]:
                        equal = False
                except KeyError:
                    equal = False

            if not equal:
                break

    return equal