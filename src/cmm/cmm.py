import numpy as np
from cmm import spectral_funcs as sf
from typing import List
from sklearn.cluster import KMeans
from cmm import utils
import jax.numpy as jnp
from jax.lax import scan
from cmm.utils import foldxy
from scipy.linalg import eigh
from cmm.utils import build_fft_trial_projection_matrices


def compute_coefs(
    xnt: np.array,
    nperseg: int,
    noverlap: int,
    fs: float,
    freq_minmax,
):

    n, t = xnt.shape

    print(f"n, t: {n, t}")
    valid_DFT_Wktf, valid_iDFT_Wktf = build_fft_trial_projection_matrices(
        t, nperseg=nperseg, noverlap=noverlap, fs=fs, freq_minmax=freq_minmax
    )
    xnkf_coefs = np.tensordot(xnt, valid_DFT_Wktf, axes=(1, 1))
    xnt_proj = jnp.einsum("ijm, jlm->il", xnkf_coefs, valid_iDFT_Wktf).real
    xntf_proj = jnp.einsum("ijm, jlm->ilm", xnkf_coefs, valid_iDFT_Wktf).real
    return xnkf_coefs, xnt_proj, xntf_proj


def compute_clusters(
    xnt: np.array,
    nperseg: int,
    noverlap: int,
    fs: float,
    freq_minmax=[-np.inf, np.inf],
):
    n, t = xnt.shape

    valid_DFT_Wktf, valid_iDFT_Wktf = build_fft_trial_projection_matrices(
        t, nperseg=nperseg, noverlap=noverlap, fs=fs, freq_minmax=freq_minmax
    )

    xnkf_coefs = np.tensordot(xnt, valid_DFT_Wktf, axes=(1, 1))
    k = xnkf_coefs.shape[1]

    pn_f = np.sqrt(jnp.einsum("ijk, ijk->ik", xnkf_coefs, np.conj(xnkf_coefs)) / k)
    xnkf_coefs_normalized = xnkf_coefs / pn_f[:, None]
    pkkf = (
        jnp.einsum(
            "ijk, ilk->jlk", xnkf_coefs_normalized, np.conj(xnkf_coefs_normalized)
        )
        / n
    )
    Vp = [eigh(m, subset_by_index=[k - 1, k - 1]) for m in pkkf.transpose([2, 0, 1])]
    eigvals_p = np.array(list(zip(*Vp))[0])
    eigvecs_p_fk = np.array(list(zip(*Vp))[1]).squeeze()
    eigvec_backproj_ft = jnp.einsum("ktf, fk->ft", valid_iDFT_Wktf, eigvecs_p_fk).real

    r = {}
    r["eigvals_p"] = eigvals_p
    r["eigvecs_p_fk"] = eigvecs_p_fk
    r["eigvec_backproj_ft"] = eigvec_backproj_ft
    r["pkkf"] = pkkf
    r["DWktf"] = valid_DFT_Wktf
    r["iDWktf"] = valid_iDFT_Wktf

    return r

    # print("pre einsum calculations")

    # DW_pkkf_tkf = jnp.einsum("ktf,klf->tlf", valid_DFT_Wktf, pkkf)
    # DW_pkkf_WD_ttf = jnp.einsum("tkf,klf->tlf", DW_pkkf_tkf, np.conj(valid_DFT_Wktf))
    # DW_pkkf_WD_ftt = np.real(DW_pkkf_WD_ttf.transpose([2, 0, 1]))
    # tt = DW_pkkf_WD_ftt.shape[-1]
    # print("finished einsum calculations")

    # V = [eigh(m, subset_by_index=[tt - 1, tt - 1]) for m in DW_pkkf_WD_ftt]
    # eigvals = np.array(list(zip(*V))[0])
    # eigvecs_ft = np.array(list(zip(*V))[1]).squeeze().real


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
