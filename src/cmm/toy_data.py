import numpy as np
from cmm.utils import build_fft_trial_projection_matrices


def make_toy_data(
    n: int,
    t: int,
    fs: float,
    m: int,
    nperseg: int,
    noverlap: float,
    noise: float,
    tau=0.1,
):

    step = nperseg - noverlap
    k = (t - noverlap) // step

    ft = nperseg // 2 + 1
    unit_circle_points = _uniform_points_on_circle((m, k, ft))
    mags_mean_f = 2 * np.exp(-np.arange(ft) / ft / tau)  # variance
    mags_mf = np.random.normal(size=(m, ft)) * np.sqrt(mags_mean_f)[None]
    # mags_mf = mags_mean_f[None]
    ymkf = mags_mf[:, None] * unit_circle_points

    xnkf = (ymkf[:,None,:,:] * _uniform_points_on_circle((n, 1, ft))).reshape([m * n, k, -1])

    valid_DFT_Wktf, valid_iDFT_Wktf = build_fft_trial_projection_matrices(
        t=t, nperseg=nperseg, noverlap=noverlap, fs=fs
    )

    xnt = np.einsum("mkf,ktf->mt", xnkf, valid_iDFT_Wktf).real
    ymt = np.einsum("mkf,ktf->mt", ymkf, valid_iDFT_Wktf).real

    # ymt = np.fft.irfft(ymkf, axis=-1).reshape([m, -1])
    # xnt = np.fft.irfft(xnkf, axis=-1).reshape([n * m, -1])
    t = xnt.shape[-1]
    xnt += np.random.randn(n * m, t) * np.sqrt(noise)
    xax = np.arange(xnt.shape[-1]) / fs
    return xnt, ymt, xax, xnkf


def _uniform_points_on_circle(size):
    angles = np.random.uniform(low=0, high=2 * np.pi, size=size)
    return np.exp(1j * angles)
