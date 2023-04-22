import os
import sys
from glob import glob
from time import time

import matplotlib.pyplot as pl
import numpy as np
import tifffile as tf
from cmm.optimize import cmm

import numpy as np

pl.style.use("dark_background")
sys.path.append("/groups/ahrens/home/ruttenv/code/zfish/")
from zfish.util import filesys as fs
import zarr
import jax

jax.config.update("jax_platform_name", "cpu")


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
# im = zarr.open(imzarr, mode="r")[: 15 * 200]


dt, dx, dy = im.shape
print(f"dt, dx, dy: {dt, dx, dy}")
fs = 15.0
nperseg = int(fs * 20)
# nperseg = int(fs * 5)
noverlap = int(0.6 * nperseg)
freq_minmax = [-np.inf, np.inf]
# freq_minmax = [1.5, 4]

xnt = im.reshape([dt, dx * dy]).T
silhouettes = []
times = []
scan = {}
opt_in_freqdom = True
# for m in range(3, 50, 10):
itemax = 1000
print_ite = 500
for m in [10]:
    t0 = time()
    print(m)
    cm = cmm(
        xnt,
        m=m,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        freq_minmax=freq_minmax,
        itemax=itemax,
        print_ite=print_ite,
        method="eigh",
        # method="svds",
        use_jax=True,
        savepath=savepath,
    )

    t1 = time()
    tt = int(t1 - t0)
    print(f"time: {tt}")
    print("starting to save results")
    # silhouettes.append(compute_silhouette(cm.coherence_mn, cm.labels))
    times.append(tt)

    # scan[m] = {}
    # scan[m]["r"] = results
    # scan[m]["times"] = times[-1]
    # np.save(savepath + "run_loop3_freqnolim", scan)
