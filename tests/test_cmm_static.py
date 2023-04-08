from cmm import spectral_funcs as sf
from cmm import utils
import matplotlib.pyplot as pl
import numpy as np
import matplotlib as mpl
from jax.lax import scan
import jax.numpy as jnp
from cmm.toy_data import make_toy_data
from cmm.cmm import compute_cluster_mean

np.random.seed(0)
mpl.pyplot.rcParams.update({"text.usetex": True})
path = "/Users/ruttenv/Documents/code/cmm/results/"

rpath = path + "res.npy"

t_ = 800
fs = 20
nperseg = 80
noverlap = int(0.8 * nperseg)
subn = 4
m = 2

#######################
### make toy data
xnt, ymt, xax = make_toy_data(subn, t_, fs, m, nperseg)
n, t = xnt.shape

#######################
### compute coherence
coh_xx, freq_ = sf.compute_coherence(
    xnt, xnt, fs=fs, nperseg=nperseg, noverlap=noverlap
)

coh_yy, freq_ = sf.compute_coherence(
    ymt, ymt, fs=fs, nperseg=nperseg, noverlap=noverlap
)

coh_yx, freq_ = sf.compute_coherence(
    ymt, xnt, fs=fs, nperseg=nperseg, noverlap=noverlap
)


freq_minmax = [-np.inf, np.inf]

rs = []
for i in range(2):
    r = compute_cluster_mean(
        xnt[(subn * i) : subn * (i + 1)],
        nperseg=nperseg,
        noverlap=noverlap,
        fs=fs,
        freq_minmax=freq_minmax,
    )
    rs.append(r)

for ind, r in enumerate(rs):
    ymt_ = np.array(r.mean(0)[None])
    coh_yx_, freq_ = sf.compute_coherence(
        ymt_, xnt, fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    pl.figure(figsize=(8, 2))
    pl.pcolormesh(coh_yx_[:, :, :10].mean(-1))
    pl.yticks([])
    pl.xlabel("observations")
    pl.title(f"coherence with learnt latent {ind}")
    pl.colorbar(orientation="horizontal")
    pl.savefig(path + f"coherence_wlatent{ind}", bbox_inches="tight")

plot = True


# pkkf, dpd, dft, mat, eigvals, eigvecs = compute_clusters(
#     xnt, nperseg=nperseg, noverlap=noverlap, fs=fs, freq_minmax=freq_minmax
# )
# r = {}
# r["pkkf"] = pkkf
# r["dpd"] = dpd
# r["dft"] = dft
# r["mat"] = mat
# r["eigvals"] = eigvals
# r["eigvecs"] = eigvecs

# np.save(rpath, r)
# r = np.load(rpath, allow_pickle=True).item()

# eigvals = r["eigvals"]  # w x t
# eigvecs = r["eigvecs"]
# ind = 1

# plot recosntructions
# nperseg = 80
# noverlap = int(0.8 * nperseg)
# freq_minmax = [-np.inf, np.inf]
# xnkf_coefs, valid_DFT_Wktf, valid_iDFT_Wktf, DFT, xnt_proj = cmm.compute_coefs(
#     xnt, nperseg=nperseg, noverlap=noverlap, fs=fs, freq_minmax=freq_minmax
# )
# ind = 5
# pl.figure(figsize=(21, 3))
# pl.plot(xnt_proj[ind], lw=5, label="backproj")
# pl.plot(xnt[ind], "-o", label="raw")
# pl.legend(loc=1)
# title_offset = 1.05
# pl.title(f"reconstruction nperseg: {nperseg}", fontsize=20, y=title_offset)
# pl.savefig(path + f"reconstruction nperseg {nperseg}", bbox_inches="tight")


if plot:
    title_offset = 1.05
    mf = 10

    ## plot data
    fig, axs = pl.subplots(nrows=m, ncols=2, figsize=(21, 4 * m), sharex=True)
    for ind in range(m):
        axs[ind, 0].plot(xax, ymt[ind], lw=4, alpha=0.8)
        axs[ind, 1].plot(xax, xnt[(ind * subn) : (ind + 1) * subn].T)
        axs[ind, 0].set_title(f"cluster mean {ind}")
        axs[ind, 1].set_title(f"observations {ind}")
    axs[-1, 0].set_xlabel("time")
    axs[-1, 1].set_xlabel("time")
    pl.savefig(path + "data", bbox_inches="tight")

    ## plot zoom on data
    fig, axs = pl.subplots(nrows=m, ncols=2, figsize=(21, 4 * m), sharex=True)
    for ind in range(m):
        axs[ind, 0].plot(xax[:nperseg], ymt[ind][:nperseg], lw=4, alpha=0.8)
        axs[ind, 1].plot(
            xax[:nperseg], xnt[(ind * subn) : (ind + 1) * subn][:, :nperseg].T
        )
        axs[ind, 0].set_title(f"cluster mean {ind}")
        axs[ind, 1].set_title(f"observations {ind}")
    axs[-1, 0].set_xlabel("time")
    axs[-1, 1].set_xlabel("time")
    pl.savefig(path + "data_zoom", bbox_inches="tight")

    ## plot coherences
    pl.figure(figsize=(5, 5))
    pl.imshow(coh_xx[:, :, :mf].mean(-1))
    ratio = 1
    pl.colorbar(fraction=0.046 * ratio, pad=0.04)
    pl.xlabel("observations")
    pl.ylabel("observations")
    pl.tight_layout()
    pl.title("coherence matrix", fontsize=20, y=title_offset)
    pl.savefig(path + "coherence_xx", bbox_inches="tight")

    pl.figure(figsize=(5, 5))
    pl.imshow(coh_yx[:, :, :mf].mean(-1))
    pl.colorbar(
        location="bottom",
    )
    pl.ylabel("cluster means")
    pl.xlabel("observations")
    pl.tight_layout()
    pl.title("cross coherence yx", fontsize=20, y=title_offset)
    pl.savefig(path + "coherence_yx", bbox_inches="tight")


# pl.figure(figsize=(21, 4))
# pl.plot(xax, xnt[:2].T)
# pl.title("data", fontsize=20)
# pl.xlabel("time")
# pl.savefig(path + "data", bbox_inches="tight")


# pl.figure(figsize=(21, 4))
# pl.plot(freqs, eigvals[:, 0])
# pl.title("top eigenvalues", fontsize=20)
# pl.xlabel("frequency")
# pl.savefig(path + "eigenvalues", bbox_inches="tight")


# pl.figure(figsize=(21, 4))
# pl.plot(eigvecs[:, 0].T)
# pl.title("top eigenvectors", fontsize=20)
# pl.xlabel("time")
# pl.savefig(path + "eigenvectors", bbox_inches="tight")


# V[ind][0][:10]
# pkkf = r["pkkf"][:, :, ind]
# dft = r["dft"][:, :, ind]

# mat = dft.T @ pkkf @ np.conj(dft)
# print(mat.imag.sum())

# print(dft.shape, pkkf.shape)


# tmp = np.allclose(pkkf[:, :, ind], np.conj(pkkf[:, :, ind].T))

# print(tmp)
# print(np.abs(pkkf[:3, :3, 0]))


# vpath = path + "V.npy"
# np.save(vpath, V)
# V = np.load(vpath)
# print(V.shape)

# pl.figure(figsize=(21, 3))
# pl.plot(V[0])
# pl.title("cluster means", fontsize=20)
# pl.savefig(path + "cluster mean", bbox_inches="tight")


# print(f"DFT: {DFT.shape}")
# print(f"pxy: {pkkf.shape}")
# print(f"DFT_Wkt: {valid_DFT_Wktf.shape}")
# print(f"k: {k}, f: {f}")


# ind = 3
# tmp = valid_DFT_Wktf[:, :, ind].T @ pkkf[:, :, ind]
# tmp2 = tmp @ valid_DFT_Wktf[:, :, ind]
# print(tmp.shape)
# print(np.allclose(tmp, DW_pkkf_tkf[:, :, ind]))
# print(np.allclose(tmp2, DW_pkkf_WD_ttf[:, :, ind]))
# print(DW_pkkf_tkf.shape)
# print(DW_pkkf_WD_ttf.shape)


# xknf_coefs_, freqs_ = sf.compute_spectral_coefs(
#     xnt,
#     fs=fs,
#     nperseg=nperseg,
#     noverlap=noverlap,
#     detrend=None,
#     freq_minmax=freq_minmax,
#     return_onesided=True,
# )
# xnkf_coefs_ = xknf_coefs_.transpose([1, 0, 2])

# print(xnkf_coefs.shape, xnkf_coefs_.shape)


# print(valid_DFT_Wktf.shape, k, t, f)


# pl.figure(figsize=(15, 15))
# # pl.imshow(np.real(DFT), aspect=1)
# ind = 4
# pl.plot(np.real(DFT[-ind]))
# pl.plot(np.real(DFT[-2]))
# pl.title("$R(\mathrm{DFT})$ waveform")
# pl.savefig(path + "DFT_waveform", bbox_inches="tight")

# pl.figure(figsize=(15, 15))
# pl.imshow(np.real(DFT), aspect=1)
# pl.title("$R(\mathrm{DFT})$ matrix")
# pl.savefig(path + "DFT", bbox_inches="tight")


# pl.figure(figsize=(15, 15))
# pl.matshow(Wkt, aspect=t / k / 3)
# pl.title("W")
# pl.savefig(path + "W", bbox_inches="tight")
