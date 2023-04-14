import jax.numpy as np
from scipy.linalg import eigh as scieigh

from cmm.utils import build_fft_trial_projection_matrices


def compute_spectral_coefs_by_hand(
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
    return xnkf_coefs


def compute_cluster_mean_minimal(
    xnt: np.array,
):
    xnkf_coefs = xnt
    n, k, f = xnkf_coefs.shape
    pn_f = np.sqrt(np.einsum("nkf, nkf->nf", xnkf_coefs, np.conj(xnkf_coefs)))
    xnkf_coefs_normalized = xnkf_coefs / pn_f[:, None]

    pkkf = np.einsum(
        "nkf, nlf->klf", xnkf_coefs_normalized, np.conj(xnkf_coefs_normalized)
    )

    Vp = [scieigh(m, subset_by_index=[k - 1, k - 1]) for m in pkkf.transpose([2, 0, 1])]
    eigvals_p = np.array(list(zip(*Vp))[0]).squeeze()
    eigvecs_p_fk = np.array(list(zip(*Vp))[1]).squeeze()
    return eigvecs_p_fk, eigvals_p


def compute_cluster_mean(
    xnt: np.array,
    nperseg: int,
    noverlap: int,
    fs: float,
    freq_minmax=[-np.inf, np.inf],
    x_in_coefs=True,  # xknf
    return_temporal_proj=True,
    normalize=True,
):
    if not x_in_coefs:
        n, t = xnt.shape

        valid_DFT_Wktf, valid_iDFT_Wktf = build_fft_trial_projection_matrices(
            t, nperseg=nperseg, noverlap=noverlap, fs=fs, freq_minmax=freq_minmax
        )
        # this does not detrendreturn_onesided=False,
        xnkf_coefs = np.tensordot(xnt, valid_DFT_Wktf, axes=(1, 1))

    else:
        xnkf_coefs = xnt
        n = xnkf_coefs.shape[0]

    k = xnkf_coefs.shape[1]
    n, k, f = xnkf_coefs.shape
    pn_f = np.sqrt(np.einsum("nkf, nkf->nf", xnkf_coefs, np.conj(xnkf_coefs)))
    if normalize:
        xnkf_coefs_normalized = xnkf_coefs / pn_f[:, None]
    else:
        xnkf_coefs_normalized = xnkf_coefs

    pkkf = np.einsum(
        "nkf, nlf->klf", xnkf_coefs_normalized, np.conj(xnkf_coefs_normalized)
    )
    Vp = [scieigh(m, subset_by_index=[k - 1, k - 1]) for m in pkkf.transpose([2, 0, 1])]
    eigvals_p = np.array(list(zip(*Vp))[0]).squeeze()
    eigvecs_p_fk = np.array(list(zip(*Vp))[1]).squeeze()
    if return_temporal_proj:
        eigvec_backproj_ft = np.einsum(
            "ktf, fk->ft", valid_iDFT_Wktf, eigvecs_p_fk
        ).real
        return eigvec_backproj_ft, eigvals_p
    else:
        return eigvecs_p_fk, eigvals_p
