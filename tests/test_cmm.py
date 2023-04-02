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

np.random.seed(3)
mpl.pyplot.rcParams.update({"text.usetex": True})
path = "/Users/ruttenv/Documents/code/cmm/results/"

rpath = path + "res.npy"

t_ = 800
fs = 20
nperseg = 80
noverlap = int(0.8 * nperseg)
subn = 5
m = 4
freq_minmax = [0, 3]
freq_minmax = [-np.inf, np.inf]
noise = 1e-3
#######################
### make toy data
xnt, ymt, xax = make_toy_data(subn, t_, fs, m, nperseg, noise)
n, t = xnt.shape

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
cm.back_proj_means()

print(f"random: {cm.labels_init}")
print(f"kmeans: {cm.kmeans.labels_}")
print(f"yours: {cm.labels}")
