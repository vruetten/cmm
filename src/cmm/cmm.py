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
from cmm.cmm_funcs import compute_cluster_mean, compute_spectral_coefs_by_hand
from cmm.spectral_funcs import compute_coherence


class CMM:
    def __init__(
        self,
        xnt: np.array,
        k: int,
        fs: float,
        nperseg: int,
        noverlap=None,
        freq_minmax=[-np.inf, np.inf],
        opt_in_freqdom=True,
    ):
        self.xnt = xnt
        self.n, self.t = xnt.shape
        self.k = k
        self.fs = fs
        self.nperseg = nperseg
        if noverlap is None:
            self.noverlap = int(0.6 * nperseg)
        else:
            self.noverlap = noverlap
        self.freq_minmax = freq_minmax
        self.opt_in_freqdom = opt_in_freqdom
        self.detrend = False
        self.initialize()

    def initialize(
        self,
    ):
        self.intialize_clusters()
        if self.opt_in_freqdom:
            self.initialize_coefs()

    def backproj_means(
        self,
    ):
        self.ymtf = self.backproject(self.coefs_ymkf)

    def analyse_results(
        self,
    ):
        # back project data
        self.xntf = self.backproject(self.coefs_xnkf)
        self.ymtf = self.backproject(self.coefs_ymkf)
        self.coherence_mnf = self.compute_cross_coherence_from_coefs(
            self.coefs_ymkf, self.coefs_xnkf
        )

    def estimate_spectrum(
        self,
        coefs_xnkf,
    ):
        pxnf, freqs = sf.estimate_spectrum(
            coefs_xnkf,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            x_in_coefs=True,
            alltoall=False,
            detrend=self.detrend,
        )
        return pxnf.real, freqs

    def backproject(self, coefs_znkf):
        return np.einsum("mkf,ktf->mtf", coefs_znkf, self.valid_iDFT_Wktf).real

    def project_to_coefs(self, xnt):
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
    ):
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
    ):
        self.kmeans = KMeans(n_clusters=self.k, random_state=0, n_init="auto").fit(
            self.xnt
        )
        self.kmeans_init_mt = self.kmeans.cluster_centers_

        self.labels_init = np.random.randint(low=0, high=self.k, size=self.n)
        self.labels = self.labels_init
        self.means_mt = self.kmeans_init_mt
        if self.opt_in_freqdom:
            self.coefs_ymkf = np.array(self.project_to_coefs(self.means_mt))
            self.f = self.coefs_ymkf.shape[-1]
            self.eigvals_kf = np.zeros([self.k, self.f])

    def get_cluster_means(
        self,
    ):
        for i in range(self.k):
            if self.opt_in_freqdom:
                valid_inds = self.labels == i
                subdata_nkf = self.coefs_xnkf[valid_inds]
                if sum(valid_inds) == 1:
                    subdata_nkf = subdata_nkf[:]
                if sum(valid_inds) == 0:
                    print("no data point allocated")
                    break
                eigvecs_fk, eigvals_f = compute_cluster_mean(
                    subdata_nkf,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    fs=self.fs,
                    freq_minmax=self.freq_minmax,
                    x_in_coefs=True,
                    return_temporal_proj=False,
                )

                self.coefs_ymkf[i] = eigvecs_fk.T
                self.eigvals_kf[i] = eigvals_f

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

    def compute_cross_coherence_from_coefs(self, coefs_ymkf, coefs_xnkf):
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
    ):
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
        self.coherence_mn = coherence_mnk.mean(-1)
        self.labels = np.argmax(self.coherence_mn, axis=0)

    def optimize(self, itemax: int):
        for ite in range(itemax):
            self.old_labels = self.labels
            self.get_cluster_means()
            self.allocate_data_to_clusters()
            if (self.labels == self.old_labels).all():
                print(f"ite: {ite} - converged")
                break
            else:
                self.old_labels = self.labels
