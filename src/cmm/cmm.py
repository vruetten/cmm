import numpy as np
from cmm import spectral_funcs as sf
from typing import List
from sklearn.cluster import KMeans
from cmm import utils
import jax.numpy as jnp
from jax.lax import scan
from cmm.utils import foldxy


def compute_clusters(
    xnt: np.array,
    nperseg: int,
    noverlap: int,
    fs: float,
    freq_minmax,
):
    n, t = xnt.shape
    Wkt = utils.make_window_matrix(t, nperseg=nperseg, noverlap=noverlap)
    DFT, freqs = utils.get_fftmat(t, fs=fs)

    valid_freq_inds = (np.abs(freqs) >= freq_minmax[0]) & (
        np.abs(freqs) <= freq_minmax[1]
    )
    valid_freqs = freqs[valid_freq_inds]
    f = len(valid_freqs)
    valid_DFT = DFT[:, valid_freq_inds]
    valid_DFT_Wktf = valid_DFT[None] * Wkt[:, :, None]
    xnkf_coefs = np.tensordot(xnt, valid_DFT_Wktf, axes=(1, 1))
    k = xnkf_coefs.shape[1]
    init = np.zeros([k, k, f]).astype("complex64")
    fxy = lambda v, y: foldxy(v, y)
    xnkf_coefs = jnp.array(xnkf_coefs)
    pkkf, _ = scan(fxy, init, (xnkf_coefs, xnkf_coefs))  # average over n
    pkkf /= n

    DW_pkkf_tkf = np.einsum("ijk,ilk->jlk", valid_DFT_Wktf, pkkf)
    DW_pkkf_WD_ttf = np.einsum("ijk,jlk->ilk", DW_pkkf_tkf, np.conj(valid_DFT_Wktf))
    DW_pkkf_WD_ftt = np.real(DW_pkkf_WD_ttf.transpose([2, 0, 1]))
    V = [np.linalg.eigh(m) for m in DW_pkkf_WD_ftt]
    eigvals = np.array(list(zip(*V))[0])[:, ::-1]
    eigvecs = np.array(list(zip(*V))[1])[:, :, ::-1]

    return pkkf, DW_pkkf_WD_ftt, valid_DFT_Wktf, DW_pkkf_WD_ftt, eigvals, eigvecs
    # return V


class CMM:
    def __init__(
        self, xnt: np.array, k: int, fs: float, nperseg: int, freq_minmax: List
    ):
        self.xnt = xnt
        self.n, self.t = xnt.shape
        self.k = k
        self.fs = fs
        self.nperseg = nperseg
        self.freq_minmax = freq_minmax

    def initialize(
        self,
    ):
        self.coefs_xknf, self.freqs = self.convert_to_spectral_coefs(self.xnt)
        self.intialize_clusters()

    def intialize_clusters(
        self,
    ):
        self.kmeans = KMeans(n_clusters=self.k, random_state=0, n_init="auto").fit(
            self.xnt
        )
        self.means_init_mt = self.kmeans.cluster_centers_
        self.labels_init = self.kmeans.labels_
        self.coefs_ykmf, _ = self.convert_to_spectral_coefs(
            self.means_init_mt
        )  # TODO: normalize to have unit norm

    def update_cluster_means(
        self,
    ):
        pass

    def allocate_data_to_clusters(
        self,
    ):
        pxyf = sf.estimate_spectrum_from_coefs(
            self.coefs_xknf, self.coefs_ykmf, fs=self.fs, abs=True
        )

        # allocations = np.argwhere(cnmf, axis=0)

    def convert_to_spectral_coefs(
        self,
        xnt,
    ):
        coefs_xknf, freqs = sf.compute_spectral_coefs(
            xnt=xnt,
            fs=self.fs,
            window="hann",
            nperseg=self.nperseg,
            freq_minamx=self.freq_minmax,
        )
        return coefs_xknf, freqs


def compute_coherence(Xnf: np.array, Ymf: np.array, timedomain=True) -> np.array:
    """Computes coherence

    Parameters:
    Xnf:
    Ymf:
    timedomain:
    """

    cnmf = np.ones()
    return cnmf
