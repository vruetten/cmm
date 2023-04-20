import os
import sys
from glob import glob
from time import time

import matplotlib.pyplot as pl
import numpy as np
import tifffile as tf
from cmm import cmm
from cmm.utils import compute_silhouette

import jax
import jax.numpy as jnp

# Global flag to set a specific platform, must be used at startup.
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_platform_name", "gpu")

pl.style.use("dark_background")
sys.path.append("/groups/ahrens/home/ruttenv/code/zfish/")
import zarr
from jax.lib import xla_bridge
from zfish.util import filesys as fs

print(f"using: {xla_bridge.get_backend().platform}")


base_dir = "/nrs/ahrens/Virginia_nrs/behavior_rig_flow/230304_f474_9dpf_casper/"
fnum = fs.get_fnum(base_dir)
print(f"fnum: {fnum}")

t0 = time()

exp = 0
dirs_ = fs.get_subfolders(base_dir)
folder_path = dirs_[f"exp{exp}"]
dirs = fs.get_subfolders(folder_path)
os.makedirs(dirs["main"] + "cmm", exist_ok=True)
os.makedirs(dirs["main"] + "cmm/results/", exist_ok=True)
dirs = fs.get_subfolders(folder_path)
savepath = dirs["cmm"] + "/results/"

impath = glob(dirs["imag_crop"] + "*.tif")[0]
imzarr = tf.imread(impath, aszarr=True)
im = zarr.open(imzarr, mode="r")[: 15 * 200]


dt, dx, dy = im.shape
print(f"dt, dx, dy: {dt, dx, dy}")
fs = 15.0
nperseg = int(fs * 20)
noverlap = int(0.6 * nperseg)
freq_minmax = [-np.inf, np.inf]
# freq_minmax = [1.5, 4]

xnt = im.reshape([dt, dx * dy]).T
silhouettes = []
times = []
scan = {}
opt_in_freqdom = True
# for m in range(3, 50, 10):
for m in [10]:
    t0 = time()
    print(m)
    cm = cmm.CMM(
        xnt,
        m=m,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        freq_minmax=freq_minmax,
        opt_in_freqdom=opt_in_freqdom,
    )

    cm.optimize(300)
    t1 = time()
    tt = int(t1 - t0)
    print(f"time: {tt}")
    print("starting to save results")
    # cm.save_results(savepath + f"run_m{m}")
    results = cm.store_results()
    results["xnt"] = 0
    silhouettes.append(compute_silhouette(cm.coherence_mn, cm.labels))
    times.append(tt)

    scan[m] = {}
    scan[m]["r"] = results
    scan[m]["silhouette"] = silhouettes[-1]
    scan[m]["times"] = times[-1]
    scan[m]["t"] = dt
    np.save(savepath + "run_loop3_freqnolim", scan)
