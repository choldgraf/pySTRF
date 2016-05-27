"""Quick utility functions."""

from IPython import embed
import numpy as np
import pandas as pd
import mne
import sys
from os import path, sep, remove
from glob import glob
from datetime import datetime


__all__ = ['bin_and_apply']


def bin_and_apply(data, bin_centers, func=np.mean):
    """Aggregate data by bin centers and apply a function to combine

    This will find the nearest bin_center for each value in data,
    group them together by the closest bin, and apply func to the
    values.

    Parameters
    ----------
    data : np.array, shape(n_points,)
        The 1-d data you'll be binning
    bin_centers : np.array, shape(n_bins)
        The centers you compare to each data point
    func : function, returns scalar from 1-d data
        Data within each bin group will be combined
        using this function.

    Returns
    -------
    new_vals : np.array, shape(n_bins)
        The data combined according to bins you supplied."""
    ix_bin = np.digitize(data, bin_centers)
    new_vals = []
    for ibin in np.unique(ix_bin):
        igroup = data[ix_bin == ibin]
        new_vals.append(func(igroup))
    new_vals = np.array(new_vals)
    return(new_vals)


def decimate_by_binning(data, data_names, n_decim):
    """Decimates along the first axis."""
    data_names = np.array(data_names).astype(int)

    # Calculate binning paramters
    n_bins = int(data.shape[0] / n_decim)
    bins = np.hstack([i]*n_decim for i in range(n_bins))

    # Do the binning and take mean between them
    data = np.vstack([data[bins == i].mean(0) for i in np.unique(bins)])
    data_names = np.array([int(data_names[bins == i].mean())
                           for i in np.unique(bins)])
    return data, data_names
