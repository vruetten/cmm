from cmm import utils
from cmm import spectral_funcs as sf
import matplotlib.pyplot as pl
import numpy as np
import matplotlib as mpl
from jax.lax import scan
import jax.numpy as jnp
from cmm.toy_data import make_toy_data
from cmm.cmm import compute_cluster_mean
from cmm import cmm

np.random.seed(4)
mpl.pyplot.rcParams.update({"text.usetex": True})
path = "/Users/ruttenv/Documents/code/cmm/results/"

rpath = path + "res.npy"

t_ = 800
fs = 20
nperseg = 80
noverlap = int(0.8 * nperseg)
subn = 4
m = 3
freq_minmax = [0, 3]
freq_minmax = [-np.inf, np.inf]
noise = 1e-4
tau = 0.1
#######################
### make toy data
xnt, ymt, xax, xknf = make_toy_data(
    subn, t_, fs, m, nperseg=nperseg, noverlap=noverlap, noise=noise, tau=tau
)
n, t = xnt.shape
print(f"n t: {n, t}")

opt_in_freqdom = False
opt_in_freqdom = True
k = m + 1
cm = cmm.CMM(
    xnt,
    k=k,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    freq_minmax=freq_minmax,
    opt_in_freqdom=opt_in_freqdom,
)

itemax = 20
cm.optimize(itemax=itemax)
if opt_in_freqdom:
    cm.backproj_means()
    coherence_mnk = cm.compute_cross_coherence_from_coefs(cm.coefs_ymkf, cm.coefs_xnkf)

tmp = np.vstack([coherence_mnk[0].sum(0), cm.eigvals_kf[0]])
print(tmp[:, :4])

print(f"random: {cm.labels_init}")
print(f"kmeans: {cm.kmeans.labels_}")
print(f"yours: {cm.labels}")

xntf_proj = cm.backproject(cm.coefs_xnkf)

fmax = cm.coefs_xnkf.shape[-1]
ind = 0
pl.figure(figsize=(21, 5))
pl.plot(xntf_proj[ind, :, :fmax].sum(-1), lw=5, label="backproj")
pl.plot(xnt[ind], "-o", label="raw")
pl.legend(loc=1)
title_offset = 1.05
pl.title(f"reconstruction nperseg: {nperseg}", fontsize=20, y=title_offset)
pl.savefig(path + f"cmm reconstruction nperseg {nperseg}", bbox_inches="tight")


fmax = 15
pl.figure()
fig, axs = pl.subplots(nrows=m, ncols=2, figsize=(21, 4 * m), sharex=True)
for ind in range(m):
    axs[ind, 0].plot(xax, cm.ymtf[ind, :, :fmax].sum(-1), lw=4, alpha=0.8)
    axs[ind, 1].plot(xax, xnt[(ind * subn) : (ind + 1) * subn].T)
    axs[ind, 0].set_title(f"cluster mean {ind}")
    axs[ind, 1].set_title(f"observations {ind}")
axs[-1, 0].set_xlabel("time")
axs[-1, 1].set_xlabel("time")
pl.savefig(path + f"results nperseg {nperseg}", bbox_inches="tight")

fmax = 10
sl = slice(nperseg * 2, nperseg * 3)
pl.figure()
fig, axs = pl.subplots(nrows=m, ncols=2, figsize=(21, 4 * m), sharex=True)
for ind in range(m):
    axs[ind, 0].plot(
        xax[sl],
        cm.ymtf[ind, :, :fmax].sum(-1)[sl],
        lw=4,
        alpha=0.8,
    )
    axs[ind, 1].plot(xax[sl], xnt[(ind * subn) : (ind + 1) * subn].T[sl])
    axs[ind, 0].set_title(f"cluster mean {ind}")
    axs[ind, 1].set_title(f"observations {ind}")
axs[-1, 0].set_xlabel("time")
axs[-1, 1].set_xlabel("time")
pl.savefig(path + f"results zoom nperseg {nperseg}", bbox_inches="tight")


### plot actually learnt clusters


fmax = slice(3, 4)
sl = slice(nperseg * 2, nperseg * 4)
pl.figure()
fig, axs = pl.subplots(nrows=k, ncols=1, figsize=(21, 4 * k), sharex=True)
for ind in range(k):
    axs[ind].set_title(f"cluster {ind}")
    subdata_ntf = xntf_proj[cm.labels == ind]
    axs[ind].plot(
        xax[sl],
        cm.ymtf[ind, sl, fmax].sum(-1),
        lw=5,
        alpha=0.8,
        label="cluster mean",
        ls="--",
    )
    axs[ind].plot(xax[sl], subdata_ntf[:, sl, fmax].sum(-1).T)
    axs[ind].legend()
axs[-1].set_xlabel("time")
# axs[-1, 1].set_xlabel("time")

pl.savefig(path + f"results backproj zoom nperseg {nperseg} ", bbox_inches="tight")
