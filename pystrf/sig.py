import numpy as np
from sklearn.preprocessing import StandardScaler
from mne.filter import low_pass_filter
import statsmodels.api as sm


__all__ = ['compress_signal']


def compress_signal(sig, kind='log', fac=None):
    '''
    Parameters
    ----------
    sig : array_like
        the signal to compress
    kind : string, one of ['log', 'exp', 'sig'], or None
        Whether we use log-compression, exponential compression,
        or sigmoidal compression. If None, then do nothing.
    fac : float, int
        The factor for the sigmoid if we're using that kind of compression
        Or the root for the exponent if exponential compression.

    Returns
    -------
    out : array, shape(sig)
        The compressed signal
    '''
    # Compression
    if kind == 'sig':
        out = sigp.sigmoid(sig, fac=fac)
    elif kind == 'log':
        out = np.log(sig)
    elif kind == 'exp':
        comp = lambda x, n: x**(1. / n)
        out = comp(sig, fac)
    elif kind is None:
        out = sig
    else:
        raise Exception('You need to specify the correct kind of compression')
    return out
