from scipy.linalg import eigh as scieigh
from cmm.utils import build_fft_trial_projection_matrices
from time import time
from scipy.sparse.linalg import svds
import numpy as np


def compute_cluster_centroid_svds(
    xnkf_coefs_normalized: np.array,
):
    n, k, f = xnkf_coefs_normalized.shape
    xfnk_coefs_normalized = xnkf_coefs_normalized.transpose([2, 0, 1])
    Vp = [svds(m, k=1) for m in xfnk_coefs_normalized]
    eigvals_p = np.array(list(zip(*Vp))[1]).squeeze()
    eigvecs_p_fk = np.array(list(zip(*Vp))[2]).squeeze()
    return eigvals_p, eigvecs_p_fk


def compute_cluster_centroid_eigh(coefs_xnkf: np.array, normalize=False):
    n, k, f = coefs_xnkf.shape
    coefs_xfkn_normalized = coefs_xnkf.transpose([2, 1, 0])
    pfkk = np.einsum(
        "fkn, fln->fkl", coefs_xfkn_normalized, np.conj(coefs_xfkn_normalized)
    )
    Vp = [scieigh(m, subset_by_index=[k - 1, k - 1]) for m in pfkk]
    eigvals_p = np.array(list(zip(*Vp))[0]).squeeze()
    eigvecs_p_fk = np.array(list(zip(*Vp))[1]).squeeze()
    return eigvals_p, eigvecs_p_fk
