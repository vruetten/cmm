from importlib import reload

import jax.numpy as jnp
import matplotlib.pyplot as pl
import numpy as np
from cmm import cmm_funcs, toy_data

reload(toy_data)
np.random.seed(4)
path = "/Users/ruttenv/Documents/code/cmm/results/"
t_ = 400
fs = 20
nperseg = 250
noverlap = int(0.8 * nperseg)
subn = 5
m = 3
tau = 0.1
xnt, ymt, xax, xnkf = toy_data.make_toy_data(
    subn, t_, fs, m, nperseg, noise=0, noverlap=noverlap, tau=0.1
)
n, t = xnt.shape


freq_minmax = [-np.inf, np.inf]
nperseg_ = 80
noverlap_ = int(0.8 * nperseg_)
valid_DFT_Wktf, valid_iDFT_Wktf = cmm_funcs.build_fft_trial_projection_matrices(
    t, nperseg=nperseg_, noverlap=noverlap_, fs=fs, freq_minmax=freq_minmax
)

xnkf_coefs = np.tensordot(xnt, valid_DFT_Wktf, axes=(1, 1))
xntf_i = jnp.einsum("nkf,ktf->ntf", xnkf_coefs, valid_iDFT_Wktf).real

n, t, f = xntf_i.shape
fmax = f
ind = 2
pl.figure(figsize=(21, 5))
pl.plot(xntf_i[ind, :, :fmax].sum(-1), lw=5, label="backproj")
pl.plot(xnt[ind], "-o", label="raw")
pl.legend(loc=1)
title_offset = 1.05
pl.title(f"reconstruction nperseg: {nperseg_}", fontsize=20, y=title_offset)
pl.savefig(path + f"reconstruction nperseg {nperseg}", bbox_inches="tight")
