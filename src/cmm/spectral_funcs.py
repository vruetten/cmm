import jax.numpy as np
from utils import _triage_segments, make_chunks
from numpy import fft as sp_fft
from jax.lax import scan
from jax.lax import broadcast


def estimate_spectrum(
    xnt: np.ndarry,
    ynt=None,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="spectrum",
    axis=-1,
    freq_minmax=[0, np.inf],
    abs=False,
    alltoall=True,  # every x with every y
    return_coefs=False,
    y_in_coefs=False,
    x_in_coefs=False,
):
    """Calculate various forms of windowed FFTs for PSD, CSD, etc.

    This is a helper function that implements the commonality between
    the stft, psd, csd, and spectrogram functions. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Parameters
    ----------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    axis : int, optional
        Axis along which the FFTs are computed; the default is over the
        last axis (i.e. ``axis=-1``).

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    Notes
    -----
    Adapted from matplotlib.mlab/scipy

    .. versionadded:: 0.1.0
    """

    if not x_in_coefs:
        if len(xnt.shape) < 2:
            xnt = broadcast(xnt, sizes=(1,))
        xn = xn.shape[0]
        coefs_xknf, freqs = compute_spectral_coefs(
            xnt=xnt,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=axis,
            freq_minmax=freq_minmax,
            nfft=nfft,
        )

    else:
        xn = xnt.shape[1]
        coefs_xknf = xnt  # assume xnt is already N x K x F
        freqs = sp_fft.rfftfreq(int(coefs_xknf.shape[-1] * 2), 1 / fs)  # to check

    if ynt is not None:
        if not y_in_coefs:
            if len(ynt.shape) < 2:
                ynt = broadcast(ynt, sizes=(1,))
            yn = ynt.shape[0]
            coefs_yknf, _ = compute_spectral_coefs(
                xnt=xnt,
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=detrend,
                return_onesided=return_onesided,
                scaling=scaling,
                axis=axis,
                freq_minmax=freq_minmax,
                nfft=nfft,
            )

        else:
            yn = ynt.shape[1]
            coefs_yknf = ynt
        if not alltoall:
            if yn != xn:
                raise Exception("invalid dimensions for not all to all")

    wn = coefs_xknf.shape[-1]
    kn = coefs_xknf.shape[0]

    if ynt is not None:
        fxy = lambda v, y: foldxy(v, y, alltoall)
        if alltoall:
            init = np.zeros([xn, yn, wn]).astype("complex64")
        else:
            init = np.zeros([xn, wn]).astype("complex64")
        pxy, _ = scan(fxy, init, (coefs_xknf, coefs_yknf))
    else:
        fxy = lambda v, y: fold(v, y, alltoall)
        if alltoall:
            init = np.zeros([xn, xn, wn]).astype("complex64")
        else:
            init = np.zeros([xn, wn]).astype("complex64")
        pxy, _ = scan(fxy, init, coefs_xknf)

    pxy /= kn  # average over k

    if abs:
        pxy = np.abs(pxy)

    return pxy, freqs


def foldxy(carry, x, alltoall):
    x0 = x[0]
    y0 = x[1]
    xn = x0.shape[0]
    yn = y0.shape[0]
    if alltoall:
        tmp0 = broadcast(x0, sizes=(yn,))
        tmp1 = broadcast(y0, sizes=(xn,))
        tmp0 = tmp0.transpose([1, 0, 2])
    else:
        tmp0 = x0
        tmp1 = y0
    carry += tmp0 * np.conj(tmp1)  # nx x ny x w
    return carry, 0


def fold(carry, x0, alltoall):
    xn = x0.shape[0]
    if alltoall:
        tmp0 = broadcast(x0, sizes=(yn,))
        tmp1 = tmp0.transpose([1, 0, 2])
    else:
        tmp0 = x0
        tmp1 = x0
    carry += tmp0 * np.conj(tmp1)  # nx x ny x w
    return carry, 0


def compute_spectral_coefs(
    xnt: np.ndarray,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    detrend="constant",
    return_onesided=True,
    scaling="spectrum",
    axis=-1,
    freq_minmax=[0, np.inf],
    nfft=None,
):
    """Compute Spectral Fourier Coefs"""

    def detrend_func(d):
        from scipy.signal import signaltools

        return signaltools.detrend(d, type=detrend, axis=-1)

    pass

    n, t = xnt.shape
    win, nperseg = _triage_segments(
        window, nperseg, input_length=t
    )  # win is a 1d array

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError("nfft must be greater than or equal to nperseg.")
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be less than nperseg.")

    if return_onesided:
        if np.iscomplexobj(x):
            sides = "twosided"
        else:
            sides = "onesided"
    else:
        sides = "twosided"

    if sides == "twosided":
        freqs = sp_fft.fftfreq(nfft, 1 / fs)
    elif sides == "onesided":
        freqs = sp_fft.rfftfreq(nfft, 1 / fs)

    if scaling == "density":
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == "spectrum":
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError("Unknown scaling: %r" % scaling)

    coefs_xnkf = myfft_helper(xnt, win, detrend_func, nperseg, noverlap, nfft, sides)
    coefs_xknf = coefs_xnkf.transpose([1, 0, 2])
    coefs_xknf *= scale

    valid_freqs = (freqs >= freq_minmax[0]) & (freqs <= freq_minmax[1])
    freqs = freqs[valid_freqs]
    coefs_xknf = coefs_xknf[:, :, valid_freqs]

    return coefs_xknf, freqs


def myfft_helper(
    x: np.ndarray,
    win: np.ndarray,
    detrend_func: function,
    nperseg: int,
    noverlap: int,
    nfft,
    sides: str,
):
    """
    Calculate windowed FFT, for internal use by
    `scipy.signal._spectral_helper`.
    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.
    Returns
    -------
    result : ndarray
        Array of FFT data
    Notes
    -----
    Adapted from matplotlib.mlab
    .. versionadded:: 0.16.0
    """
    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        result = make_chunks(x, nperseg, noverlap)

    # Detrend each data segment individually
    result = detrend_func(result)  # default to last axis - result is N x K x T

    # Apply window by multiplication

    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == "twosided":
        func = sp_fft.fft
    else:
        result = (
            result.real
        )  # forces result to be real (should be real anyway as doing one sided fft...)
        func = sp_fft.rfft
    coefs_nkf = func(result, n=nfft)

    return coefs_nkf
