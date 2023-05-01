import numpy as np
from cmm.utils import build_fft_trial_projection_matrices, get_freqs
from sklearn.cluster import KMeans
from cmm.spectral_funcs import compute_coherence
from time import time
from cmm.cmm_funcs import (
    compute_cluster_centroid_svds,
    compute_cluster_centroid_eigh,
)
from functools import partial

np.random.seed(0)


def cmm(
    xnt: np.array,
    m: int,
    fs: float,
    nperseg: int,
    noverlap: int,
    freq_minmax=[0, np.inf],
    itemax=1000,
    print_ite=10,
    method="eigh",
    savepath=None,
    use_jax=True,
    xy=None,
):
    print(f"using method: {method}")
    print(f"using jax: {use_jax}")
    n, t = xnt.shape
    valid_DFT_Wktf, valid_iDFT_Wktf = build_fft_trial_projection_matrices(
        t, nperseg=nperseg, noverlap=noverlap, fs=fs, freq_minmax=freq_minmax
    )
    coefs_xnkf = np.tensordot(xnt, valid_DFT_Wktf, axes=(1, 1))
    coefs_xfkn = coefs_xnkf.transpose([2, 1, 0])
    pf_n = np.sqrt(np.einsum("fkn, fkn->fn", coefs_xfkn, np.conj(coefs_xfkn)))
    coefs_xnkf_normalized = (coefs_xfkn / pf_n[:, None]).transpose([2, 1, 0])

    full_freqs, freqs, valid_freq_inds = get_freqs(nperseg, fs, freq_minmax)
    labels = np.random.randint(low=0, high=m, size=n)

    kmeans = KMeans(n_clusters=m, random_state=0, n_init="auto").fit(xnt)
    means_mt = kmeans.cluster_centers_
    coefs_ymkf = np.tensordot(means_mt, valid_DFT_Wktf, axes=(1, 1))
    f = coefs_ymkf.shape[-1]

    if method == "svds":
        print("using svds")
        compute_cluster_mean = compute_cluster_centroid_svds
    else:
        print("using eigh")
        compute_cluster_mean = partial(compute_cluster_centroid_eigh, use_jax=use_jax)

    def compute_cross_coherence_from_coefs(
        coefs_ymkf: np.array, coefs_xnkf: np.array
    ) -> np.array:
        coherence_mnk, _ = compute_coherence(
            coefs_ymkf,
            coefs_xnkf,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            freq_minmax=freq_minmax,
            x_in_coefs=True,
            y_in_coefs=True,
            detrend=False,
        )
        return coherence_mnk

    def allocate_data_to_clusters(coefs_ymkf: np.array, coefs_xnkf: np.array):
        coherence_mnk = compute_cross_coherence_from_coefs(coefs_ymkf, coefs_xnkf)
        coherence_mn = coherence_mnk.mean(-1)
        labels = np.argmax(coherence_mn, axis=0)
        return labels

    def compute_cluster_centroids(coefs_xnkf_normalized, labels):
        eigvals = []
        eigvecs = []
        for ind, i in enumerate(np.unique(labels)):
            valid_inds = labels == i
            subdata_nkf = coefs_xnkf_normalized[valid_inds]
            if sum(valid_inds) == 0:
                print("no data point allocated")
                break
            eigvals_f, eigvecs_fk = compute_cluster_mean(subdata_nkf)
            eigvals.append(eigvals_f)
            eigvecs.append(eigvecs_fk)
        return np.array(eigvals), np.array(eigvecs).transpose([0, 2, 1])

    t0 = time()
    for ite in range(itemax):
        if np.mod(ite, print_ite) == 0:
            t1 = time()
            diff = np.round((t1 - t0) / 60, 2)
            print(f"at ite: {ite}, time: {diff}mins")
        old_labels = labels
        eigvals, coefs_ymkf = compute_cluster_centroids(coefs_xnkf_normalized, labels)
        labels = allocate_data_to_clusters(coefs_ymkf, coefs_xnkf)
        if (labels == old_labels).all():
            print(f"ite: {ite} - converged")
            break
        else:
            old_labels = labels
    coherence_mnf = compute_cross_coherence_from_coefs(coefs_ymkf, coefs_xnkf)
    xax = np.arange(t)
    xmin = xax / 60
    if xy is None:
        labels_im = None
    else:
        labels_im = labels.reshape([xy[0], xy[1]])

    r = {}
    r["m"] = m
    r["fs"] = fs
    r["nperseg"] = nperseg
    r["noverlap"] = noverlap
    r["freq_minmax"] = freq_minmax

    r["freqs"] = freqs
    r["full_freqs"] = full_freqs
    r["valid_freq_inds"] = valid_freq_inds
    # r["xnt"] = xnt
    r["coefs_ymkf"] = coefs_ymkf
    r["coefs_xnkf"] = coefs_xnkf
    r["xax"] = xax
    r["xmin"] = xmin

    r["coherence_mnf"] = coherence_mnf
    r["labels"] = labels
    r["valid_iDFT_Wktf"] = valid_iDFT_Wktf
    r["valid_DFT_Wktf"] = valid_DFT_Wktf
    r["labels_im"] = labels_im
    # if savepath is not None:
    #     np.save(savepath, r)

    return r
