from time import time
import numpy as np
from sklearn.cluster import KMeans
from cmm import spectral_funcs as sf
from cmm.cmm_funcs import compute_cluster_centroid_eigh
from cmm.spectral_funcs import compute_coherence
from cmm.utils import build_fft_trial_projection_matrices, get_freqs
# from cmm.utils import compute_silhouette
from cmm.utils import timeit
import jax.numpy as jnp
from multiprocessing import Pool
from importlib import reload
import dask.array as da
reload(sf)

np.random.seed(0)


class CMM:
    def __init__(
        self,
        data_txy: np.array,
        m: int,
        fs: float,
        nperseg: int,
        noverlap=None,
        freq_minmax=[0, np.inf],
        normalize=True,
        savepath=None,
        nprocesses=None,
        random_state=0,
    ):
        if len(data_txy.shape) == 3:
            self.im = True
            self.dxdy = data_txy.shape[1:]
            x, y = self.dxdy
            self.xnt = data_txy.reshape([-1, x * y]).T
        elif len(data_txy.shape) == 2:
            self.im = False
            self.xnt = data_txy
            self.dxdy = None

        if freq_minmax[-1] <= freq_minmax[0]:
            freq_minmax[1] = np.inf
            print("freq minmax reset")

        self.n, self.t = self.xnt.shape
        self.m = m
        self.fs = fs
        self.xax = np.arange(self.t) / self.fs
        self.xmin = self.xax / 60
        self.nperseg = nperseg
        self.random_state = random_state
        if noverlap is None:
            self.noverlap = int(0.6 * nperseg)
        else:
            self.noverlap = noverlap
        self.freq_minmax = freq_minmax
        self.detrend = False
        self.normalize = normalize
        self.savepath = savepath

        self.nprocesses = nprocesses
        self.initialize()

        if self.freq_minmax[0] < 0:
            self.freq_minmax[0] = 0
        if self.freq_minmax[1] == 0:
            self.freq_minmax[1] = np.inf

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

    def estimate_spectrum(self, coefs_xnkf, x_in_coefs=True, alltoall = False, abs = True):
        pxnf, freqs = sf.estimate_spectrum(
            coefs_xnkf,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            x_in_coefs=x_in_coefs,
            alltoall=alltoall,
            abs=abs,
            detrend=self.detrend,
            t = self.t,
        )
        if abs:
            return pxnf.real, freqs
        else:
            return pxnf, freqs

    def initialize_coefs(
        self,
    ) -> None:
        self.coefs_xnkf = self.project_to_coefs(self.xnt)
        coefs_xfkn = self.coefs_xnkf.transpose([2, 1, 0])
        pf_n = np.sqrt(jnp.einsum("fkn, fkn->fn", coefs_xfkn, np.conj(coefs_xfkn)))
        self.coefs_xnkf_normalized = (coefs_xfkn / pf_n[:, None]).transpose([2, 1, 0]) # normalize by total amount of power at said frequency

        self.full_freqs, self.freqs, self.valid_freq_inds = get_freqs(
            self.nperseg, self.fs, self.freq_minmax
        )
        self.xax = np.arange(self.t) / self.fs

    def intialize_clusters(
        self,
    ) -> None:
        self.kmeans = KMeans(n_clusters=self.m, random_state=self.random_state).fit(
            self.xnt
        )
        self.kmeans_init_mt = self.kmeans.cluster_centers_
        self.labels_init = self.kmeans.labels_
        self.labels = self.labels_init
        self.means_mt = self.kmeans_init_mt
        self.coefs_ymkf = np.array(self.project_to_coefs(self.means_mt))
        self.f = self.coefs_ymkf.shape[-1]
        self.eigvals_mf = np.zeros([self.m, self.f])

    def get_cluster_centroids(
        self,
    ) -> None:
        print("get cluster centroids")
        for i in range(self.m):
            valid_inds = self.labels == i
            # print(f"label shape: {self.labels.shape}")
            # print(self.labels)
            # print(valid_inds)
            ncells = sum(valid_inds)
            if ncells> 0:
                print(f"cluster {i} has {ncells} data points")
                subdata_nkf = self.coefs_xnkf_normalized[valid_inds]
                eigvals_f, eigvecs_fk = compute_cluster_centroid_eigh(subdata_nkf)
                self.coefs_ymkf[i] = eigvecs_fk.T
                self.eigvals_mf[i] = eigvals_f
            else:
                print("no data point allocated")


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
            t = self.t,
        )
        return coherence_mnk

    def compute_auto_coherence_from_coefs(self, coefs_xnkf, normalize = True) -> np.array:
        n = coefs_xnkf.shape[0]
        xax = np.arange(n)
        pxx_nnf, _ = sf.estimate_spectrum(
            coefs_xnkf,
            fs=self.fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            freq_minmax=self.freq_minmax,
            x_in_coefs=True,
            detrend=self.detrend,
            alltoall=True,
            abs=True,
            t = self.t,
        )
        diag = da.diagonal(pxx_nnf, offset=0, axis1=0, axis2=1).T
        if normalize:
            coherence_nnf = pxx_nnf**2 / (diag[:, None] * diag[None])
        else:
            normalization = (diag[:, None] * diag[None]).sum(-1)
            coherence_nnf = pxx_nnf**2/normalization[:,:,None]
        return coherence_nnf

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
            t = self.t,
        )

        return pxyf, freqs

    def allocate_data_to_clusters(
        self, average = True
    ) -> None:
        coherence_mnk = self.compute_cross_coherence_from_coefs(
            self.coefs_ymkf, self.coefs_xnkf
        )
        if average:
            self.coherence_mn = coherence_mnk.mean(-1)  # average coherence over frequencies
            self.labels = np.argmax(self.coherence_mn, axis=0).compute()
        else:
            self.labels = np.argmax(self.coherence_mnk, axis=0).compute()

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
                self.get_cluster_centroids() # the first round
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
        ymt_coh = []
        ymt_all = []
        ymt_coh_all = []
        for ind, l in enumerate(np.unique(self.labels)):
            weighted_iDTF_ktf = np.array(
                self.valid_iDFT_Wktf * self.pxf[self.labels == l].mean(0)
            )
            coherence_weighted_iDTF_ktf = np.array(
                self.valid_iDFT_Wktf * self.coherence_mnf[l, self.labels == l].mean(0)
            )

            if minmax is not None:
                vfreqs = np.argwhere(
                    (freqs > minmax[0]) & (freqs < minmax[1])
                ).squeeze()
                notidx = [i for i in range(len(freqs)) if i not in vfreqs]
                weighted_iDTF_ktf[:, :, notidx] = 0
                freqs[notidx] = 0
            tmp = jnp.einsum(
                    "mkf,ktf->mtf", self.coefs_ymkf[l : l + 1], weighted_iDTF_ktf
                ).real
            ymt_all.append(tmp)   
            ymt.append(tmp.mean(-1))
            
            tmp = jnp.einsum(
                    "mkf,ktf->mtf",
                    self.coefs_ymkf[l : l + 1],
                    coherence_weighted_iDTF_ktf,
                ).real
            ymt_coh_all.append(tmp)
            ymt_coh_all.append(tmp.mean(-1))

        ymt = np.array(ymt).squeeze()
        ymt_coh = np.array(ymt_coh).squeeze()
        ymt_all = np.array(ymt_all).squeeze()
        # ymt_coh_all = ymt_coh_all

        return ymt, ymt_coh, ymt_all, ymt_coh_all, freqs

    def compute_model_silhouette(self, sample_size = None):
        from sklearn.metrics import silhouette_score
        coherence_nn = self.compute_auto_coherence_from_coefs(self.coefs_xnkf).mean(-1) # averages over frequencies
        silhouette = silhouette_score(coherence_nn, self.labels, metric="precomputed", sample_size=sample_size)
        return silhouette

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
        self.ymt, self.ymt_coh, self.ymt_all, self.ymt_coh_all, _ = self.backproject_centroids()
        
        
        (
            self.angles_mkfxy,
            self.angles_mfxy_mean,
            self.angles_mfxy_std,
            ) = compute_average_phase_shift(
                self.coefs_xnkf, self.coefs_ymkf
            )
        if self.im:
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
        t0 = time()
        # print("computing silhouette")
        # self.silhouette = self.compute_model_silhouette(compute_full_coherence=False)

        # self.silhouette_proxy = self.compute_model_proxy_silhouette()
        timeit(t0)

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
        r["im"] = self.im
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
        r["ymt_coh"] = self.ymt_coh
        r["convergence_ite"] = self.convergence_ite
        r["optimization_time"] = self.optimization_time
        if not hasattr(self, "coherence_mnf"):
            self.coherence_mnf = self.compute_cross_coherence_from_coefs(
                self.coefs_ymkf, self.coefs_xnkf
            )
        r["coherence_mnf"] = self.coherence_mnf
        r["labels"] = self.labels
        r["cluster_coherence_m2f"] = self.cluster_coherence_m2f
        if self.im:
            r["labels_im"] = self.labels_im
            r["angles_mf_im_mean"] = self.angles_mfxy_mean
            r["dxdy"] = self.dxdy
            r["valid_clusters"] = self.valid_clusters
            r["labels_im_valid"] = self.labels_im_valid
            # r["angles_mf_im_std"] = self.angles_mfxy_std
        if hasattr(self, "silhouette"):
            r["silhouette"] = self.silhouette
        if hasattr(self, "silhouette_proxy"):
            r["silhouette_proxy"] = self.silhouette_proxy
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


def phase_shift_cluster(angles_mfxy_mean: np.array, xy = True) -> np.array:
    if xy:
        offset = np.percentile(angles_mfxy_mean, q=50, axis=(-2, -1))
        return np.angle(
            np.exp(1j * angles_mfxy_mean) * np.exp(-1j * offset)[:, :, None, None]
        )
    else:
        offset = np.percentile(angles_mfxy_mean, q=50, axis=(-1))
        return np.angle(
            np.exp(1j * angles_mfxy_mean) * np.exp(-1j * offset)[:, :, None]
        )


def compute_cluster_coherence(coherence_mnf: np.array, labels: np.array) -> np.array:
    cluster_coherence = []
    m, n, f = coherence_mnf.shape
    for ind, l in enumerate(np.unique(labels)):
        cluster_coherence.append(coherence_mnf[ind, labels == l].mean(0))
        cluster_coherence.append(coherence_mnf[ind, labels != l].mean(0))
    cluster_coherence_m2f = np.array(cluster_coherence).reshape([-1, 2, f])
    return cluster_coherence_m2f


def compute_average_phase_shift(
    coefs_xnkf: np.array, coefs_ymkf: np.array, x=None, y=None, center=True
) -> np.array:
    angles_mnkf = np.angle(coefs_ymkf[:, None] * np.conj(coefs_xnkf[None]))
    m, n, k, f = angles_mnkf.shape
    if x is not None:
        axis = 1
        angles_mkfxy = angles_mnkf.reshape([m, x, y, k, f]).transpose([0, 3, 4, 1, 2])
        angles_mfxy_mean = np.mean(angles_mkfxy, axis=axis)
        angles_mfxy_std = np.std(np.abs(angles_mkfxy), axis=axis)/np.sqrt(k)
        if center:
            angles_mfxy_mean = phase_shift_cluster(angles_mfxy_mean, xy=True)
    else:
        axis = 2
        angles_mkfxy = angles_mnkf
        angles_mfxy_mean = np.mean(angles_mnkf, axis=axis)
        angles_mfxy_std = np.std(angles_mnkf, axis=axis)/np.sqrt(k)
        
        if center:
            angles_mfxy_mean = phase_shift_cluster(angles_mfxy_mean, xy=False)

    
    return angles_mkfxy, angles_mfxy_mean, angles_mfxy_std
