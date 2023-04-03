import jax.numpy as np
from cmm.utils import build_fft_trial_projection_matrices2
from scipy.linalg import eigh


def compute_spectral_coefs_by_hand(
    xnt: np.array,
    nperseg: int,
    noverlap: int,
    fs: float,
    freq_minmax=[-np.inf, np.inf],
):
    n, t = xnt.shape
    valid_DFT_Wktf, valid_iDFT_Wktf = build_fft_trial_projection_matrices2(
        t, nperseg=nperseg, noverlap=noverlap, fs=fs, freq_minmax=freq_minmax
    )

    xnkf_coefs = np.tensordot(xnt, valid_DFT_Wktf, axes=(1, 1))

    return xnkf_coefs


def compute_cluster_mean(
    xnt: np.array,
    nperseg: int,
    noverlap: int,
    fs: float,
    freq_minmax=[-np.inf, np.inf],
    x_in_coefs=False,  # xknf
    return_temporal_proj=True,
):
    if not x_in_coefs:
        n, t = xnt.shape

        valid_DFT_Wktf, valid_iDFT_Wktf = build_fft_trial_projection_matrices2(
            t, nperseg=nperseg, noverlap=noverlap, fs=fs, freq_minmax=freq_minmax
        )
        # this does not detrendreturn_onesided=False,
        xnkf_coefs = np.tensordot(xnt, valid_DFT_Wktf, axes=(1, 1))

    else:
        xnkf_coefs = xnt
        n = xnkf_coefs.shape[0]

    k = xnkf_coefs.shape[1]
    pn_f = np.sqrt(
        np.einsum("ijk, ijk->ik", xnkf_coefs, np.conj(xnkf_coefs)) / k
    )  # TODO: check that you're supposed to divide by k
    xnkf_coefs_normalized = xnkf_coefs / pn_f[:, None]
    pkkf = (
        np.einsum(
            "ijk, ilk->jlk", xnkf_coefs_normalized, np.conj(xnkf_coefs_normalized)
        )
        / n
    )

    Vp = [eigh(m, subset_by_index=[k - 1, k - 1]) for m in pkkf.transpose([2, 0, 1])]
    eigvals_p = np.array(list(zip(*Vp))[0])
    eigvecs_p_fk = np.array(list(zip(*Vp))[1]).squeeze()
    if return_temporal_proj:
        eigvec_backproj_ft = np.einsum(
            "ktf, fk->ft", valid_iDFT_Wktf, eigvecs_p_fk
        ).real
        return eigvec_backproj_ft
    else:
        return eigvecs_p_fk
