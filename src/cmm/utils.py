import numpy as np
from numpy import fft as sp_fft
from typing import Tuple


def timeit(t0):
    from time import time

    t = time() - t0
    print(f"time :{np.round(t, 5)}")


def compute_avg_clust_dist(cluster: np.array):
    n = cluster.shape[0]
    clust_dist = np.abs((cluster[None] - cluster[:, None]))
    clust_avg_dist = np.mean(clust_dist[np.triu_indices(n)])
    return clust_avg_dist


def compute_silhouette(coherence_mn: np.array, labels: np.array):
    labels_unique = np.unique(labels)
    m, n = coherence_mn.shape
    silhouette = 0
    for i in labels_unique:
        inds = labels == i
        ninds = labels != i
        if sum(inds) == 0:
            print("no elements in cluster - skip")
        elif sum(ninds) == 0:
            print("all elements in this cluster")
        else:
            inclust = coherence_mn[i, inds]
            inclust_avg_dist = compute_avg_clust_dist(inclust)
            outclust = coherence_mn[i, ninds]
            outclust_avg_dist = compute_avg_clust_dist(outclust)
            silhouette += (outclust_avg_dist - inclust_avg_dist) / np.max(
                [outclust_avg_dist, inclust_avg_dist]
            )
    silhouette /= m
    return silhouette


def build_DFT_matrix(t, f, real=True):
    if real:
        if np.mod(f, 2) == 0:
            ff = np.arange((f + 2) // 2)
        else:
            ff = np.arange((f + 1) // 2)
        DFT_tf = np.exp(-1j * 2 * np.pi * np.arange(t)[None] * ff[:, None] / t).T
    else:
        ff = np.arange(f)
        DFT_tf = np.exp(-1j * 2 * np.pi * np.arange(t)[None] * ff[:, None] / t).T
    return DFT_tf


def build_fft_trial_projection_matrices(
    t: int,
    nperseg: int,
    noverlap: int,
    fs=1,
    freq_minmax=[-np.inf, np.inf],
    win_type="hann",
):
    from scipy.signal.windows import get_window

    win = get_window(win_type, nperseg)  # return a 1d array
    step = nperseg - noverlap
    k = (t - noverlap) // step

    win_kt = np.repeat(win[None], axis=0, repeats=k)
    scale = 1.0 / win.sum() ** 2

    DFT_ff = build_DFT_matrix(nperseg, nperseg, real=True) * scale
    wd_tf = win[:, None] * DFT_ff
    iwd_tf = win[:, None] * 1 / DFT_ff / nperseg

    f = DFT_ff.shape[1]
    Wktf = np.zeros(shape=(k, t, f)) * 1j
    iWktf = np.zeros(shape=(k, t, f)) * 1j
    Wktf[0, :nperseg] = wd_tf
    iWktf[0, :nperseg] = iwd_tf

    for i in range(1, k):
        Wktf[i, i * step : i * step + nperseg] = wd_tf
        iWktf[i, i * step : i * step + nperseg] = iwd_tf

    return Wktf, iWktf


def foldxy(carry, x):
    import jax.numpy as jnp
    from jax.lax import broadcast

    x0 = x[0]
    y0 = x[1]
    xn = x0.shape[0]
    yn = y0.shape[0]
    tmp0 = broadcast(x0, sizes=(yn,))
    tmp1 = broadcast(y0, sizes=(xn,))
    tmp0 = tmp0.transpose([1, 0, 2])
    carry += tmp0 * jnp.conj(tmp1)  # k x k' x w
    return carry, 0


def get_fftmat(t: int, fs: 1) -> Tuple[np.array, np.array]:
    dftmtx = np.fft.fft(np.eye(t))
    freqs = np.fft.fftfreq(t, d=1 / fs)
    return dftmtx, freqs


def make_chunks(x: np.ndarray, nperseg: int, noverlap: int):
    """rechuncks data efficiently using strides"""
    step = nperseg - noverlap
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result  # result is N x K x T


def _triage_segments(window, nperseg: int, input_length: int):
    from scipy.signal.windows import get_window
    import warnings

    """
    Parses window and nperseg arguments for spectrogram and _spectral_helper.
    This is a helper function, not meant to be called externally.
    Parameters
    ----------
    window : string, tuple, or ndarray # default is "hann"
        If window is specified by a string or tuple and nperseg is not
        specified, nperseg is set to the default of 256 and returns a window of
        that length.
        If instead the window is array_like and nperseg is not specified, then
        nperseg is set to the length of the window. A ValueError is raised if
        the user supplies both an array_like window and a value for nperseg but
        nperseg does not equal the length of the window.
    nperseg : int
        Length of each segment
    input_length: int
        Length of input signal, i.e. x.shape[-1]. Used to test for errors.
    Returns
    -------
    win : ndarray
        window. If function was called with string or tuple than this will hold
        the actual array used as a window.
    nperseg : int
        Length of each segment. If window is str or tuple, nperseg is set to
        256. If window is array_like, nperseg is set to the length of the
        window.
    """
    # parse window; if array like, then set nperseg = win.shape
    if isinstance(window, str) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            warnings.warn(
                "nperseg = {0:d} is greater than input length "
                " = {1:d}, using nperseg = {1:d}".format(nperseg, input_length)
            )
            nperseg = input_length
        win = get_window(window, nperseg)  # return a 1d array
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError("window must be 1-D")
        if input_length < win.shape[-1]:
            raise ValueError("window is longer than input signal")
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError(
                    "value specified for nperseg is different" " from length of window"
                )
    return win, nperseg
