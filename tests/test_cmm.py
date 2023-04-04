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
subn = 5
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

itemax = 2
cm.optimize(itemax=itemax)
if opt_in_freqdom:
    cm.backproj_means()

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


fmax = 20
pl.figure()
fig, axs = pl.subplots(nrows=m, ncols=2, figsize=(21, 4 * m), sharex=True)
for ind in range(m):
    axs[ind, 0].plot(xax, cm.ymtf[ind, :, :].sum(-1), lw=4, alpha=0.8)
    axs[ind, 1].plot(xax, xnt[(ind * subn) : (ind + 1) * subn].T)
    axs[ind, 0].set_title(f"cluster mean {ind}")
    axs[ind, 1].set_title(f"observations {ind}")
axs[-1, 0].set_xlabel("time")
axs[-1, 1].set_xlabel("time")
pl.savefig(path + f"results nperseg {nperseg}", bbox_inches="tight")


sl = slice(nperseg, nperseg * 2)
pl.figure()
fig, axs = pl.subplots(nrows=m, ncols=2, figsize=(21, 4 * m), sharex=True)
for ind in range(m):
    axs[ind, 0].plot(
        xax[sl],
        cm.ymtf[ind, :, :20].sum(-1)[sl],
        lw=4,
        alpha=0.8,
    )
    axs[ind, 1].plot(xax[sl], xnt[(ind * subn) : (ind + 1) * subn].T[sl])
    axs[ind, 0].set_title(f"cluster mean {ind}")
    axs[ind, 1].set_title(f"observations {ind}")
axs[-1, 0].set_xlabel("time")
axs[-1, 1].set_xlabel("time")
pl.savefig(path + f"results zoom nperseg {nperseg}", bbox_inches="tight")
