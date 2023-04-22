import jax.numpy as jnp
from scipy.linalg import eigh as scieigh
from cmm.utils import build_fft_trial_projection_matrices
from cmm import power_iteration as pi
from scipy.sparse.linalg import svds


def compute_spectral_coefs_by_hand(
    xnt: jnp.array,
    nperseg: int,
    noverlap: int,
    fs: float,
    freq_minmax=[0, jnp.inf],
):
    n, t = xnt.shape
    valid_DFT_Wktf, valid_iDFT_Wktf = build_fft_trial_projection_matrices(
        t, nperseg=nperseg, noverlap=noverlap, fs=fs, freq_minmax=freq_minmax
    )
    xnkf_coefs = jnp.tensordot(xnt, valid_DFT_Wktf, axes=(1, 1))
    return xnkf_coefs


def compute_cluster_mean_minimal_fast(
    xnkf_coefs_normalized: jnp.array,
):
    n, k, f = xnkf_coefs_normalized.shape
    xfnk_coefs_normalized = xnkf_coefs_normalized.transpose([2, 0, 1])
    Vp = [svds(m, k=1) for m in xfnk_coefs_normalized]
    eigvals_p = jnp.array(list(zip(*Vp))[1]).squeeze()
    eigvecs_p_fk = jnp.array(list(zip(*Vp))[2]).squeeze()
    return eigvecs_p_fk, eigvals_p


def compute_cluster_mean_minimal_power_ite(
    xnkf_coefs_normalized: jnp.array,
):
    from jax import vmap

    n, k, f = xnkf_coefs_normalized.shape
    xfnk_coefs_normalized = xnkf_coefs_normalized.transpose([2, 0, 1])
    eigvals_p, eigvecs_p_fk = vmap(pi.power_iteration_jit)(xfnk_coefs_normalized)
    return eigvecs_p_fk, eigvals_p


def compute_cluster_mean_minimal(coefs_xnkf: jnp.array, normalize=False):
    n, k, f = coefs_xnkf.shape
    if normalize:
        xfkn_coefs = coefs_xnkf.transpose([2, 1, 0])
        pf_n = jnp.sqrt(jnp.einsum("fkn, fkn->fn", xfkn_coefs, jnp.conj(xfkn_coefs)))
        coefs_xfkn_normalized = xfkn_coefs / pf_n[:, None]
    else:
        coefs_xfkn_normalized = coefs_xnkf.transpose([2, 1, 0])

    pfkk = jnp.einsum(
        "fkn, fln->fkl", coefs_xfkn_normalized, jnp.conj(coefs_xfkn_normalized)
    )
    Vp = [scieigh(m, subset_by_index=[k - 1, k - 1]) for m in pfkk]
    eigvals_p = jnp.array(list(zip(*Vp))[0]).squeeze()
    eigvecs_p_fk = jnp.array(list(zip(*Vp))[1]).squeeze()
    return eigvecs_p_fk, eigvals_p


def compute_cluster_mean(
    xnt: jnp.array,
    nperseg: int,
    noverlap: int,
    fs: float,
    freq_minmax=[-jnp.inf, jnp.inf],
    x_in_coefs=True,  # xknf
    return_temporal_proj=True,
    normalize=True,
):
    if not x_in_coefs:
        n, t = xnt.shape

        valid_DFT_Wktf, valid_iDFT_Wktf = build_fft_trial_projection_matrices(
            t, nperseg=jnp.rseg, noverlap=noverlap, fs=fs, freq_minmax=freq_minmax
        )
        # this does not detrendreturn_onesided=False,
        xnkf_coefs = jnp.tensordot(xnt, valid_DFT_Wktf, axes=(1, 1))

    else:
        xnkf_coefs = xnt
        n = xnkf_coefs.shape[0]

    k = xnkf_coefs.shape[1]
    n, k, f = xnkf_coefs.shape
    pn_f = jnp.sqrt(jnp.einsum("nkf, nkf->nf", xnkf_coefs, jnp.conj(xnkf_coefs)))
    if normalize:
        xnkf_coefs_normalized = xnkf_coefs / pn_f[:, None]
    else:
        xnkf_coefs_normalized = xnkf_coefs

    pkkf = jnp.einsum(
        "nkf, nlf->klf", xnkf_coefs_normalized, jnp.conj(xnkf_coefs_normalized)
    )
    Vp = [scieigh(m, subset_by_index=[k - 1, k - 1]) for m in pkkf.transpose([2, 0, 1])]
    eigvals_p = jnp.array(list(zip(*Vp))[0]).squeeze()
    eigvecs_p_fk = jnp.array(list(zip(*Vp))[1]).squeeze()
    if return_temporal_proj:
        eigvec_backproj_ft = jnp.einsum(
            "ktf, fk->ft", valid_iDFT_Wktf, eigvecs_p_fk
        ).real
        return eigvec_backproj_ft, eigvals_p
    else:
        return eigvecs_p_fk, eigvals_p
