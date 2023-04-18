from time import time

import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans

from cmm import spectral_funcs as sf
from cmm.cmm_funcs import (
    compute_cluster_mean,
    compute_cluster_mean_minimal,
    compute_spectral_coefs_by_hand,
)
from cmm.spectral_funcs import compute_coherence
from cmm.utils import build_fft_trial_projection_matrices


class CMM:
    def __init__(
        self,
        xnt: np.array,
        m: int,
        fs: float,
        nperseg: int,
        noverlap=None,
        freq_minmax=[-np.inf, np.inf],
        opt_in_freqdom=True,
        normalize=True,
        savepath=None,
    ):
        self.xnt = xnt
        self.n, self.t = xnt.shape
        self.m = m
        self.fs = fs
        self.xax = np.arange(self.t) / self.fs
        self.xmin = self.xax / 60
        self.nperseg = nperseg
        if noverlap is None:
            self.noverlap = int(0.6 * nperseg)
        else:
            self.noverlap = noverlap
        self.freq_minmax = freq_minmax
        self.opt_in_freqdom = opt_in_freqdom
        self.detrend = False
        self.normalize = normalize
        self.savepath = savepath
        self.initialize()

    def initialize(
        self,
    ):
        self.intialize_clusters()
        if self.opt_in_freqdom:
            self.initialize_coefs()
            self.k = self.coefs_xnkf.shape[1]

    def save_results(self, savepath=None):
        if savepath is None:
            if self.savepath is not None:
                savepath = self.savepath
            else:
                print("need path to save to")
                return None
        r = self.store_results()
        np.save(savepath, r)
        print(f"data saved at {savepath}")

    def store_results(
        self,
    ):
        r = {}
        r["m"] = self.m
        r["fs"] = self.fs
        r["nperseg"] = self.nperseg
        r["noverlap"] = self.noverlap
        r["freq_minmax"] = self.freq_minmax
        r["xax"] = self.xax
        r["xmin"] = self.xmin
        r["freq"] = self.freq
        r["xnt"] = self.xnt
        r["coefs_ymkf"] = self.coefs_ymkf
        r["coefs_xnkf"] = self.coefs_xnkf
        if not hasattr(self, "coherence_mnf"):
            self.coherence_mnf = self.compute_cross_coherence_from_coefs(
                self.coefs_ymkf, self.coefs_xnkf
            )
        r["coherence_mnf"] = self.coherence_mnf
        r["labels"] = self.labels
        r["valid_iDFT_Wktf"] = self.valid_iDFT_Wktf
        r["valid_DFT_Wktf"] = self.valid_DFT_Wktf
        return r

    def backproj_means(
        self,
    ):
        self.ymtf = self.backproject(self.coefs_ymkf)

    def analyse_results(
        self,
    ) -> None:
        # back project data
        self.xntf = self.backproject(self.coefs_xnkf)
        self.ymtf = self.backproject(self.coefs_ymkf)
        self.coherence_mnf = self.compute_cross_coherence_from_coefs(
            self.coefs_ymkf, self.coefs_xnkf
        )

    def estimate_spectrum(self, coefs_xnkf, x_in_coefs=True):
        pxnf, freqs = sf.estimate_spectrum(
            coefs_xnkf,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            x_in_coefs=x_in_coefs,
            alltoall=False,
            detrend=self.detrend,
        )
        return pxnf.real, freqs

    def backproject(self, coefs_znkf) -> np.array:
        return jnp.einsum("mkf,ktf->mtf", coefs_znkf, self.valid_iDFT_Wktf).real

    def project_to_coefs(self, xnt) -> np.array:
        coefs_xnkf = compute_spectral_coefs_by_hand(
            xnt=xnt,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            freq_minmax=self.freq_minmax,
        )
        return coefs_xnkf

    def initialize_coefs(
        self,
    ) -> None:
        self.coefs_xnkf = self.project_to_coefs(self.xnt)

        (
            self.valid_DFT_Wktf,
            self.valid_iDFT_Wktf,
        ) = build_fft_trial_projection_matrices(
            self.t,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            fs=self.fs,
            freq_minmax=self.freq_minmax,
        )
        self.freq = np.fft.rfftfreq(self.nperseg, d=1 / self.fs)

    def intialize_clusters(
        self,
    ) -> None:
        self.kmeans = KMeans(n_clusters=self.m, random_state=0, n_init="auto").fit(
            self.xnt
        )
        self.kmeans_init_mt = self.kmeans.cluster_centers_

        self.labels_init = np.random.randint(low=0, high=self.m, size=self.n)
        self.labels = self.labels_init
        self.means_mt = self.kmeans_init_mt
        if self.opt_in_freqdom:
            self.coefs_ymkf = np.array(self.project_to_coefs(self.means_mt))
            self.f = self.coefs_ymkf.shape[-1]
            self.eigvals_mf = np.zeros([self.m, self.f])

    def get_cluster_means(
        self,
    ) -> None:
        for i in range(self.m):
            if self.opt_in_freqdom:
                valid_inds = self.labels == i
                subdata_nkf = self.coefs_xnkf[valid_inds]
                if sum(valid_inds) == 0:
                    print("no data point allocated")
                    break
                eigvecs_fk, eigvals_f = compute_cluster_mean_minimal(subdata_nkf)
                self.coefs_ymkf[i] = eigvecs_fk.T
                self.eigvals_mf[i] = eigvals_f

            else:
                subdata_nt = self.xnt[self.labels == i]
                eigvec_backproj_ft, eigvals_f = compute_cluster_mean(
                    subdata_nt,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    fs=self.fs,
                    freq_minmax=self.freq_minmax,
                    x_in_coefs=False,
                    return_temporal_proj=True,
                )

                self.means_mt[i] = eigvecs_fk.mean(0)

    def compute_cross_coherence_from_coefs(self, coefs_ymkf, coefs_xnkf) -> np.array:
        coherence_mnk, _ = compute_coherence(
            coefs_ymkf,
            coefs_xnkf,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            freq_minmax=self.freq_minmax,
            x_in_coefs=True,
            y_in_coefs=True,
            detrend=self.detrend,
        )
        return coherence_mnk

    def compute_crossspectrum_from_coefs(self, coefs_ymkf, coefs_xnkf):
        pxyf, freqs = sf.estimate_spectrum(
            coefs_ymkf,
            coefs_xnkf,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            freq_minmax=self.freq_minmax,
            x_in_coefs=True,
            y_in_coefs=True,
            detrend=self.detrend,
            abs=True,
        )

        return pxyf, freqs

    def allocate_data_to_clusters(
        self,
    ) -> None:
        if self.opt_in_freqdom:
            coherence_mnk = self.compute_cross_coherence_from_coefs(
                self.coefs_ymkf, self.coefs_xnkf
            )

        else:
            coherence_mnk, _ = compute_coherence(
                self.means_mt,
                self.xnt,
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                freq_minmax=self.freq_minmax,
                x_in_coefs=False,
                y_in_coefs=False,
                detrend=self.detrend,
            )
        self.coherence_mn = coherence_mnk.mean(-1)  # average coherence over frequencies
        self.labels = np.argmax(self.coherence_mn, axis=0)

    def optimize(self, itemax: int, print_ite=10) -> None:
        t0 = time()
        for ite in range(itemax):
            if np.mod(ite, print_ite) == 0:
                t1 = time()
                diff = np.round((t1 - t0) / 60, 2)
                print(f"at ite: {ite}, time: {diff}mins")
            self.old_labels = self.labels
            self.get_cluster_means()
            self.allocate_data_to_clusters()
            if (self.labels == self.old_labels).all():
                print(f"ite: {ite} - converged")
                break
            else:
                self.old_labels = self.labels


def compute_cluster_coherence(coherence_mnf, labels) -> np.array:
    cluster_coherence = []
    m, n, f = coherence_mnf.shape
    for l in np.unique(labels):
        cluster_coherence.append(coherence_mnf[l, labels == l].mean(0))
        cluster_coherence.append(coherence_mnf[l, labels != l].mean(0))
    cluster_coherence_m2f = np.array(cluster_coherence).reshape([m, 2, f])
    return cluster_coherence_m2f


def compute_average_phase_shift(
    coefs_xnkf: np.array, coefs_ymkf: np.array, x: int, y: int
) -> np.array:
    angles_mnkf = np.angle(coefs_xnkf[None] * np.conj(coefs_ymkf[:, None]))
    m, n, k, f = angles_mnkf.shape
    if x is not None:
        axis = 1
        angles_mxykf = angles_mnkf.reshape([m, x, y, k, f]).transpose([0, 3, 4, 1, 2])
    angles_mfxy_mean = np.mean(angles_mxykf, axis=axis)
    angles_mfxy_std = np.std(angles_mxykf, axis=axis)
    return angles_mxykf, angles_mfxy_mean, angles_mfxy_std
