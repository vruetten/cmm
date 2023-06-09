import jax.numpy as jnp
from cmm.utils import _triage_segments, make_chunks
from numpy import fft as sp_fft
from jax.lax import broadcast
from cmm.utils import build_fft_trial_projection_matrices


def compute_spectral_coefs(  # used in coherence
    xnt: jnp.ndarray,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    detrend="constant",
    return_onesided=True,
    scaling="spectrum",
    axis=-1,
    freq_minmax=[0, jnp.inf],
    nfft=None,
):
    """Compute Spectral Fourier Coefs"""

    def detrend_func(d):
        from scipy.signal import signaltools

        return signaltools.detrend(d, type=detrend, axis=-1)

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
        if jnp.iscomplexobj(xnt):
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

    if detrend is False:
        detrend_func = None
    coefs_xnkf = myfft_helper(xnt, win, detrend_func, nperseg, noverlap, nfft, sides)
    coefs_xnkf *= scale

    valid_freqs = (jnp.abs(freqs) >= freq_minmax[0]) & (
        jnp.abs(freqs) <= freq_minmax[1]
    )
    freqs = freqs[valid_freqs]
    coefs_xnkf = coefs_xnkf[:, :, valid_freqs]

    return coefs_xnkf, freqs


def myfft_helper(
    x: jnp.ndarray,
    win: jnp.ndarray,
    detrend_func,
    nperseg: int,
    noverlap: int,
    nfft,
    sides: str,
):
    """
    Calculate windowed FFT, for internal use by
    `scipy.signal._spectral_helper`.
    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All ijnp.t validation is performed there, and the
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
        result = x[..., jnp.newaxis]
    else:
        result = make_chunks(x, nperseg, noverlap)

    # Detrend each data segment individually
    if detrend_func is not None:
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


def compute_coherence(
    xnt: jnp.array,
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
    freq_minmax=[0, jnp.inf],
    alltoall=True,  # every x with every y
    y_in_coefs=False,
    x_in_coefs=False,
):
    pxx_nk, freq = estimate_spectrum(
        xnt=xnt,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        freq_minmax=freq_minmax,
        abs=True,
        alltoall=False,
        x_in_coefs=x_in_coefs,
    )
    pyy_mk, freq = estimate_spectrum(
        xnt=ynt,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        freq_minmax=freq_minmax,
        abs=True,
        alltoall=False,
        x_in_coefs=y_in_coefs,
    )
    pxy_nmk, freq = estimate_spectrum(
        xnt=xnt,
        ynt=ynt,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        axis=axis,
        freq_minmax=freq_minmax,
        abs=True,
        alltoall=True,
        x_in_coefs=x_in_coefs,
        y_in_coefs=y_in_coefs,
    )
    coherence_xyf = pxy_nmk**2 / (pxx_nk[:, None] * pyy_mk[None])
    return coherence_xyf, freq


def estimate_spectrum(
    xnt: jnp.array,
    ynt=None,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="spectrum",  # default scaling from scipy
    axis=-1,
    freq_minmax=[0, jnp.inf],
    abs=False,
    alltoall=True,  # every x with every y
    return_coefs=False,
    y_in_coefs=False,
    x_in_coefs=False,
    normalize_per_trial=False,
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
        xn = xnt.shape[0]
        coefs_xnkf, freqs = compute_spectral_coefs(
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
        xn = xnt.shape[0]
        coefs_xnkf = xnt  # assume xnt is already N x K x F
        f = coefs_xnkf.shape[-1]
        freqs = sp_fft.rfftfreq(f * 2 - 1, d=1 / fs)  # to check

    if ynt is not None:
        if not y_in_coefs:
            if len(ynt.shape) < 2:
                ynt = broadcast(ynt, sizes=(1,))
            yn = ynt.shape[0]
            coefs_ynkf, _ = compute_spectral_coefs(
                xnt=ynt,
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
            yn = ynt.shape[0]
            coefs_ynkf = ynt
        if not alltoall:
            if yn != xn:
                raise Exception("invalid dimensions for not all to all")

    wn = coefs_xnkf.shape[-1]
    kn = coefs_xnkf.shape[1]

    if ynt is not None:
        if alltoall:
            pxy = jnp.einsum("nkf, mkf-> nmf", coefs_xnkf, jnp.conj(coefs_ynkf))
        else:
            pxy = jnp.einsum("nkf, nkf-> nf", coefs_xnkf, jnp.conj(coefs_ynkf))

    else:
        if alltoall:
            pxy = jnp.einsum("nkf, mkf-> nmf", coefs_xnkf, jnp.conj(coefs_xnkf))
        else:
            pxy = jnp.einsum("nkf, nkf-> nf", coefs_xnkf, jnp.conj(coefs_xnkf))

    if abs:
        pxy = jnp.abs(pxy).real
    if normalize_per_trial:
        pxy /= kn

    return pxy, freqs
