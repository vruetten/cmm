import numpy as np


def make_toy_data(n: int, t: int, fs: float, m: int, nperseg: int, noise: float):
    k = t // nperseg
    ft = int((t / 2) // k)

    angles_mkf = np.random.uniform(low=0, high=2 * np.pi, size=(m, k, ft))
    mags_mean_f = 10 * np.exp(-np.arange(ft) / ft / 0.2)  # variance
    mags_mf = (np.random.normal(size=(m, ft)) * np.sqrt(mags_mean_f)[None]) ** 2

    ymkf = mags_mf[:, None] * np.exp(1j * angles_mkf)

    angles_nkf = np.random.uniform(low=0, high=2 * np.pi, size=(n, 1, ft))
    xnkf = []
    for i in range(m):
        xnkf.append(ymkf[i] * np.exp(1j * angles_nkf))
    xnkf = np.array(xnkf).reshape([m * n, k, -1])
    xnt = np.fft.irfft(xnkf, axis=-1).reshape([n * m, -1])
    t = xnt.shape[-1]
    xnt += np.random.randn(n * m, t) * np.sqrt(noise)
    ymt = np.fft.irfft(ymkf, axis=-1).reshape([m, -1])
    xax = np.arange(xnt.shape[-1]) / fs
    return xnt, ymt, xax
