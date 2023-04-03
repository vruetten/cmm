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
        self.ymtf = np.einsum("mkf,ktf->mtf", self.coefs_ymkf, self.valid_iDFT_Wktf)

    def initialize_coefs(
        self,
    ):
        # self.coefs_xknf, self.freqs = sf.compute_spectral_coefs(
        #     xnt=self.xnt,
        #     fs=self.fs,
        #     nperseg=self.nperseg,
        #     noverlap=self.noverlap,
        #     freq_minmax=self.freq_minmax,
        #     return_onesided=False,
        # )
        self.coefs_xnkf = compute_spectral_coefs_by_hand(
            xnt=self.xnt,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            freq_minmax=self.freq_minmax,
        )
        self.valid_DFT_Wktf, self.valid_iDFT_Wktf = build_fft_trial_projection_matrices(
            self.nperseg,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            fs=self.fs,
            freq_minmax=self.freq_minmax,
        )

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
            self.coefs_ymkf = np.array(
                compute_spectral_coefs_by_hand(
                    xnt=self.means_mt,
                    fs=self.fs,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    freq_minmax=self.freq_minmax,
                )
            )
            # self.coefs_ykmf, self.freqs = sf.compute_spectral_coefs(
            #     xnt=self.means_mt,
            #     fs=self.fs,
            #     nperseg=self.nperseg,
            #     noverlap=self.noverlap,
            #     freq_minmax=self.freq_minmax,
            #     return_onesided=False,
            # )

    def get_cluster_means(
        self,
    ):
        for i in range(self.k):
            if self.opt_in_freqdom:
                valid_inds = self.labels == i
                # subdata_nkf = self.coefs_xknf.transpose(1, 0, 2)[valid_inds]
                subdata_nkf = self.coefs_xnkf[valid_inds]
                if sum(valid_inds) == 1:
                    subdata_nkf = subdata_nkf[:]
                if sum(valid_inds) == 0:
                    print("no data point allocated")
                    break

                # self.coefs_ykmf[:, i] = compute_cluster_mean(
                #     subdata_nkf,
                #     nperseg=self.nperseg,
                #     noverlap=self.noverlap,
                #     fs=self.fs,
                #     freq_minmax=self.freq_minmax,
                #     x_in_coefs=True,
                #     return_temporal_proj=False,
                # ).T
                self.coefs_ymkf[i] = compute_cluster_mean(
                    subdata_nkf,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    fs=self.fs,
                    freq_minmax=self.freq_minmax,
                    x_in_coefs=True,
                    return_temporal_proj=False,
                ).T

            else:
                subdata_nt = self.xnt[self.labels == i]
                self.means_mt[i] = compute_cluster_mean(
                    subdata_nt,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    fs=self.fs,
                    freq_minmax=self.freq_minmax,
                    x_in_coefs=False,
                    return_temporal_proj=True,
                ).mean(0)

    def allocate_data_to_clusters(
        self,
    ):
        if self.opt_in_freqdom:
            coherence_mnk, _ = compute_coherence(
                self.coefs_ymkf.transpose([1, 0, 2]),
                self.coefs_xnkf.transpose([1, 0, 2]),
                fs=self.fs,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                freq_minmax=self.freq_minmax,
                x_in_coefs=True,
                y_in_coefs=True,
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
            )
        self.coherence_mn = coherence_mnk.mean(-1)
        # print(self.coherence_mn)
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
