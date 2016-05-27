import numpy as np
from .. import delay_timeseries
from numpy.testing import assert_almost_equal, assert_equal


def test_delay_timeseries():
    """Test for the custom time series lag function."""
    times = np.arange(1000).reshape([1, -1])
    sfreq = 1000
    delays = [0, .1, .2, .3]
    delayed_times = delay_timeseries(times, sfreq, delays)
    assert_equal(times[0], delayed_times[0])
    for i, idel in enumerate(delays[1:]):
        assert_equal(times[0, :-sfreq*idel], delayed_times[i+1, sfreq*idel:])
