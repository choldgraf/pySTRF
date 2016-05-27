"""Useful statistical functions."""

import numpy as np
from scipy import stats, linalg
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.preprocessing import scale


__all__ = ['partial_corr',
           'coh_to_bits',
           'snr_epochs',
           'calculate_upper_bound']


def partial_corr(C, do_scale=False):
    """
    Returns the sample linear partial correlation coefficients between pairs
    of variables in C, controlling for the remaining variables in C.

    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken
        as a variable
    do_scale : bool
        Whether to scale each column of C to mean==0 and variance==1
        before calculations

    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j]
        controlling for the remaining variables in C.

    Information
    -----------
    Partial Correlation in Python (clone of Matlab's partialcorr)
    This uses the linear regression approach to compute the partial 
    correlation (might be slow for a huge number of variables). The 
    algorithm is detailed here:
        http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    Taking X and Y two variables of interest and Z the matrix with all the
    variable minus {X, Y}, the algorithm can be summarized as
        1) perform a normal linear least-squares regression with X as the
            target and Z as the predictor
        2) calculate the residuals in Step #1
        3) perform a normal linear least-squares regression with Y as the
            target and Z as the predictor
        4) calculate the residuals in Step #3
        5) calculate the correlation coefficient between the residuals from
            Steps #2 and #4
        The result is the partial correlation between X and Y while
            controlling for the effect of Z

    Date: Nov 2014
    Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    Testing: Valentina Borghesani, valentinaborghesani@gmail.com
    URL: https://gist.github.com/fabianp/9396204419c7b638d38f
    """

    C = np.asarray(C)
    if do_scale is True:
        C = scale(C, axis=0)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def coh_to_bits(coh):
    """Convert coherence values to a measure of bits."""
    return -np.log2(1-coh)


def snr_epochs(epochs, n_perm=10, fmin=1, fmax=300, tmin=None, tmax=None,
               kind='coh', normalize_coherence=False):
    '''
    Computes the coherence between the mean of subsets of epochs. This can
    be used to assess signal stability in response to a stimulus (repeated or
    otherwise).

    Parameters
    ----------
    epochs : instance of Epochs
        The data on which to calculate coherence. Coherence will be calculated
        between the mean of random splits of trials
    n_perm : int
        The number of permuatations to run
    fmin : float
        The minimum coherence frequency
    fmax : float
        The maximum coherence frequency
    tmin : float
        Start time for coherence estimation
    tmax : float
        Stop time for coherence estimation
    kind : 'coh' | 'corr'
        Specifies the similarity statistic.
        If corr, calculate correlation between the mean of subsets of epochs.
        If coh, then calculate the coherence.
    normalize_coherence : bool
        If True, subtract the grand mean coherence across permutations and
        channels from the output matrix. This is a way to "baseline" your
        coherence values to show deviations from the global means

    Outputs
    -------
    permutations : np.array, shape (n_perms, n_signals, n_freqs)
        A collection of coherence values for each permutation.

    coh_freqs : np.array, shape (n_freqs,)
        The frequency values in the coherence analysis
    '''
    sfreq = epochs.info['sfreq']
    epochs = epochs.crop(tmin, tmax, copy=True)
    nep, n_chan, ntime = epochs._data.shape

    # Run permutations
    permutations = []
    for iperm in tqdm(xrange(n_perm)):
        # Split our epochs into two random groups, take mean of each
        t1, t2 = np.split(np.random.permutation(np.arange(nep)), [nep/2.])
        mn1, mn2 = [epochs[this_ixs]._data.mean(0)
                    for this_ixs in [t1, t2]]

        # Now compute similarity between the two
        this_similarity = []
        for ch, this_mean1, this_mean2 in zip(epochs.ch_names, mn1, mn2):
            this_means = np.vstack([this_mean1, this_mean2])
            if kind == 'coh':
                this_means = this_means[np.newaxis, :, :]
                ixs = ([0], [1])
                sim, coh_freqs, _, _, _ = spectral_connectivity(
                    this_means, sfreq=sfreq, method='coh', fmin=fmin,
                    fmax=fmax, tmin=tmin, tmax=tmax, indices=ixs, verbose=0)
                sim = sim.squeeze()
            elif kind == 'corr':
                sim, _ = pearsonr(*this_means)
            else:
                raise ValueError('Unknown similarity type: {0}'.format(kind))
            this_similarity.append(sim)
        permutations.append(this_similarity)
    permutations = np.array(permutations)

    if normalize_coherence is True:
        # Normalize coherence values to their grand average
        permutations -= permutations.mean((0, 1))

    if kind == 'coh':
        return permutations, coh_freqs
    elif kind == 'corr':
        return permutations


def calculate_upper_bound(similarity, n_ep):
    """Calculate an upper bound on model performance.

    Parameters
    ----------
    similarity : array, any shape
        The estimated similarity across epochs as estimated by snr_epochs. This
        is either the correlation or coherence between means of subsets of
        trials for one electrode.
    n_ep : int
        The number of epochs used in the coherence estimation

    Returns
    -------
    upper_bound : array, shape == similarity.shape
        The upper bound on model performance as estimated by coherence across
        trials. This is either the expected coherence or expected correlation,
        depending on which type of input is given in similarity.
    """
    right_hand = .5 * (-n_ep + n_ep * np.sqrt(1. / similarity))
    upper_bound = 1. / (right_hand + 1)
    return upper_bound
