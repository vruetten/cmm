from time import time
import numpy as np
from sklearn.cluster import KMeans
from cmm import spectral_funcs as sf
from cmm.cmm_funcs import compute_cluster_centroid_eigh
from cmm.spectral_funcs import compute_coherence
from cmm.utils import build_fft_trial_projection_matrices, get_freqs
import jax.numpy as jnp
from multiprocessing import Pool

np.random.seed(0)


class CMM:
    def __init__(
        self,
        xnt: np.array,
        m: int,
        fs: float,
        nperseg: int,
        noverlap=None,
        freq_minmax=[0, np.inf],
        normalize=True,
        savepath=None,
        dxdy=None,
        nprocesses=None,
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
        self.detrend = False
        self.normalize = normalize
        self.savepath = savepath
        self.dxdy = dxdy
        self.nprocesses = nprocesses
        self.initialize()

        if self.freq_minmax[0] < 0:
            self.freq_minmax[0] = 0
        if self.freq_minmax[1] == 0:
            self.freq_minmax[1] = np.inf
        if min(self.dxdy) == 0:
            self.dxdxy = None

    def initialize(
        self,
    ):
        self.xnt = self.xnt - self.xnt.mean(-1)[:, None]
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
        self.intialize_clusters()
        self.initialize_coefs()
        self.k = self.coefs_xnkf.shape[1]

    def backproj_means(
        self,
    ):
        self.ymtf = self.backproject(self.coefs_ymkf)

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

    def initialize_coefs(
        self,
    ) -> None:
        self.coefs_xnkf = self.project_to_coefs(self.xnt)
        coefs_xfkn = self.coefs_xnkf.transpose([2, 1, 0])
        pf_n = np.sqrt(np.einsum("fkn, fkn->fn", coefs_xfkn, np.conj(coefs_xfkn)))
        self.coefs_xnkf_normalized = (coefs_xfkn / pf_n[:, None]).transpose([2, 1, 0])

        self.full_freqs, self.freqs, self.valid_freq_inds = get_freqs(
            self.nperseg, self.fs, self.freq_minmax
        )

    def intialize_clusters(
        self,
    ) -> None:
        self.kmeans = KMeans(n_clusters=self.m, random_state=0, n_init="auto").fit(
            self.xnt
        )
        self.kmeans_init_mt = self.kmeans.cluster_centers_

        self.labels_init = np.random.randint(
            low=0,
            high=self.m,
            size=self.n,
        )
        self.labels = self.labels_init
        self.means_mt = self.kmeans_init_mt
        self.coefs_ymkf = np.array(self.project_to_coefs(self.means_mt))
        self.f = self.coefs_ymkf.shape[-1]
        self.eigvals_mf = np.zeros([self.m, self.f])

    def get_cluster_centroids(
        self,
    ) -> None:
        for i in range(self.m):
            valid_inds = self.labels == i
            subdata_nkf = self.coefs_xnkf_normalized[valid_inds]
            if sum(valid_inds) == 0:
                print("no data point allocated")
                break

            eigvals_f, eigvecs_fk = compute_cluster_centroid_eigh(subdata_nkf)
            self.coefs_ymkf[i] = eigvecs_fk.T
            self.eigvals_mf[i] = eigvals_f

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
        coherence_mnk = self.compute_cross_coherence_from_coefs(
            self.coefs_ymkf, self.coefs_xnkf
        )
        self.coherence_mn = coherence_mnk.mean(-1)  # average coherence over frequencies
        self.labels = np.argmax(self.coherence_mn, axis=0)

    def optimize(self, itemax: int, print_ite=10) -> None:
        t0 = time()
        if self.nprocesses is not None:
            pool = Pool(processes=self.nprocesses)
        for ite in range(itemax):
            if np.mod(ite, print_ite) == 0:
                t1 = time()
                diff = np.round((t1 - t0) / 60, 2)
                print(f"at ite: {ite}, time: {diff}mins")
            old_labels = self.labels
            if self.nprocesses is None:
                self.get_cluster_centroids()
            else:
                subdata = [
                    self.coefs_xnkf_normalized[self.labels == ind]
                    for ind in range(self.m)
                ]
                results = pool.map(compute_cluster_centroid_eigh, subdata)
                for ind, r in enumerate(results):
                    self.coefs_ymkf[ind] = r[1].T
                    self.eigvals_mf[ind] = r[0]

            self.allocate_data_to_clusters()
            if (self.labels == old_labels).all():
                print(f"ite: {ite} - converged - all labels stablilized")
                self.convergence_ite = ite
                t1 = time()
                diff = np.round((t1 - t0) / 60, 2)
                self.optimization_time = diff
                break
            else:
                old_labels = self.labels
        self.convergence_ite = ite
        t1 = time()
        diff = np.round((t1 - t0) / 60, 2)
        self.optimization_time = diff

    def backproject(self, coefs_znkf) -> np.array:
        return np.einsum("mkf,ktf->mtf", coefs_znkf, self.valid_iDFT_Wktf).real

    def project_to_coefs(self, xnt) -> np.array:
        coefs_xnkf = np.tensordot(xnt, self.valid_DFT_Wktf, axes=(1, 1))

        return coefs_xnkf

    def backproject_centroids(self, minmax=None):
        self.pxf, freqs = self.estimate_spectrum(self.coefs_xnkf)
        ymt = []
        for ind, l in enumerate(np.unique(self.labels)):
            weighted_iDTF_ktf = np.array(
                self.valid_iDFT_Wktf * self.pxf[self.labels == l].mean(0)
            )
            if minmax is not None:
                vfreqs = np.argwhere(
                    (freqs > minmax[0]) & (freqs < minmax[1])
                ).squeeze()
                notidx = [i for i in range(len(freqs)) if i not in vfreqs]
                weighted_iDTF_ktf[:, :, notidx] = 0
                freqs[notidx] = 0
            ymt.append(
                np.einsum(
                    "mkf,ktf->mtf", self.coefs_ymkf[l : l + 1], weighted_iDTF_ktf
                ).real.mean(-1)
            )
        ymt = np.array(ymt).squeeze()
        return ymt, freqs

    def analyse_results(
        self,
    ) -> None:
        # back project data
        # self.xntf = self.backproject(self.coefs_xnkf)
        # self.ymtf = self.backproject(self.coefs_ymkf)
        self.coherence_mnf = self.compute_cross_coherence_from_coefs(
            self.coefs_ymkf, self.coefs_xnkf
        )
        self.cluster_coherence_m2f = compute_cluster_coherence(
            self.coherence_mnf, self.labels
        )
        self.ymt, _ = self.backproject_centroids()
        if self.dxdy is not None:
            (
                self.angles_mkfxy,
                self.angles_mfxy_mean,
                self.angles_mfxy_std,
            ) = compute_average_phase_shift(
                self.coefs_xnkf, self.coefs_ymkf, self.dxdy[0], self.dxdy[1]
            )
            self.labels_im = self.labels.reshape([self.dxdy[0], self.dxdy[1]])
            self.valid_clusters, self.labels_im_valid = threshold_clusters(
                self.labels_im, self.cluster_coherence_m2f, threshold=0.7
            )
        self.silhouette = self.compute_model_proxy_silhouette()

    def compute_model_silhouette(
        self,
    ) -> np.array:
        from .utils import compute_silhouette

        coherence_nn = self.compute_cross_coherence_from_coefs(
            self.coefs_xnkf, self.coefs_xnkf
        )
        silhouette = compute_silhouette(coherence_nn=coherence_nn, labels=self.labels)
        return silhouette

    def compute_model_proxy_silhouette(
        self,
    ) -> np.array:
        from .utils import compute_silhouette_proxy

        silhouette = compute_silhouette_proxy(
            coherence_mn=self.coherence_mn, labels=self.labels
        )
        return silhouette

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
        r["freqs"] = self.freqs
        r["full_freqs"] = self.full_freqs
        r["valid_freq_inds"] = self.valid_freq_inds
        r["ymt"] = self.ymt
        r["convergence_ite"] = self.convergence_ite
        r["optimization_time"] = self.optimization_time
        # r["coefs_ymkf"] = self.coefs_ymkf
        # r["coefs_xnkf"] = self.coefs_xnkf
        if not hasattr(self, "coherence_mnf"):
            self.coherence_mnf = self.compute_cross_coherence_from_coefs(
                self.coefs_ymkf, self.coefs_xnkf
            )
        r["coherence_mnf"] = self.coherence_mnf
        r["labels"] = self.labels
        # r["valid_iDFT_Wktf"] = self.valid_iDFT_Wktf
        # r["valid_DFT_Wktf"] = self.valid_DFT_Wktf
        r["cluster_coherence_m2f"] = self.cluster_coherence_m2f
        if self.dxdy is not None:
            r["labels_im"] = self.labels_im
            r["angles_mf_im_mean"] = self.angles_mfxy_mean
            r["dxdy"] = self.dxdy
            r["valid_clusters"] = self.valid_clusters
            r["labels_im_valid"] = self.labels_im_valid
            # r["angles_mf_im_std"] = self.angles_mfxy_std
        if hasattr(self, "silhouette"):
            r["silhouette"] = self.silhouette
        return r

    def save_results(self, r=None, savepath=None):
        if savepath is None:
            if self.savepath is not None:
                savepath = self.savepath
            else:
                print("need path to save to")
                return None
        if r is None:
            r = self.store_results()
        else:
            np.save(savepath, r)
        print(f"data saved at {savepath}")


def threshold_clusters(
    labels_im: np.array, cluster_coherence_m2f: np.array, threshold: float
):
    valid_clusters = np.argwhere(
        (cluster_coherence_m2f[:, 0].max(-1) > threshold)
    ).squeeze()
    mask = np.isin(labels_im, valid_clusters, invert=True)
    labels_im_valid = labels_im.copy().astype("float")
    labels_im_valid[mask] = np.nan
    return valid_clusters, labels_im_valid


def phase_shift_cluster(angles_mfxy_mean: np.array) -> np.array:
    offset = np.percentile(angles_mfxy_mean, q=50, axis=(-2, -1))
    return np.angle(
        np.exp(1j * angles_mfxy_mean) * np.exp(-1j * offset)[:, :, None, None]
    )


def compute_cluster_coherence(coherence_mnf: np.array, labels: np.array) -> np.array:
    cluster_coherence = []
    m, n, f = coherence_mnf.shape
    for ind, l in enumerate(np.unique(labels)):
        cluster_coherence.append(coherence_mnf[ind, labels == l].mean(0))
        cluster_coherence.append(coherence_mnf[ind, labels != l].mean(0))
    cluster_coherence_m2f = np.array(cluster_coherence).reshape([m, 2, f])
    return cluster_coherence_m2f


def compute_average_phase_shift(
    coefs_xnkf: np.array, coefs_ymkf: np.array, x: int, y: int, center=True
) -> np.array:
    angles_mnkf = np.angle(coefs_ymkf[:, None] * np.conj(coefs_xnkf[None]))
    m, n, k, f = angles_mnkf.shape
    if x is not None:
        axis = 1
        angles_mkfxy = angles_mnkf.reshape([m, x, y, k, f]).transpose([0, 3, 4, 1, 2])
    angles_mfxy_mean = np.mean(angles_mkfxy, axis=axis)
    angles_mfxy_std = np.std(angles_mkfxy, axis=axis)
    if center:
        angles_mfxy_mean = phase_shift_cluster(angles_mfxy_mean)
    return angles_mkfxy, angles_mfxy_mean, angles_mfxy_std
