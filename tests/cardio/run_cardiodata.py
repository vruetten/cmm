import os
import sys
from glob import glob
from time import time

import matplotlib.pyplot as pl
import numpy as np
import tifffile as tf
from cmm import cmm

pl.style.use("dark_background")
sys.path.append("/groups/ahrens/home/ruttenv/code/zfish/")
import zarr
from jax.lib import xla_bridge
from zfish.util import filesys as fs

print(xla_bridge.get_backend().platform)


base_dir = "/nrs/ahrens/Virginia_nrs/behavior_rig_flow/230304_f474_9dpf_casper/"
fnum = fs.get_fnum(base_dir)
print(f"fnum: {fnum}")

t0 = time()

exp = 0
dirs_ = fs.get_subfolders(base_dir)
folder_path = dirs_[f"exp{exp}"]
dirs = fs.get_subfolders(folder_path)
os.makedirs(dirs["main"] + "cmm", exist_ok=True)
dirs = fs.get_subfolders(folder_path)
savepath = dirs["cmm"]

impath = glob(dirs["imag_crop"] + "*.tif")[0]
imzarr = tf.imread(impath, aszarr=True)
im = zarr.open(imzarr, mode="r")[: 15 * 100]

dt, dx, dy = im.shape
fs = 15.0
nperseg = int(fs * 20)
noverlap = int(0.6 * nperseg)
freq_minmax = [-np.inf, np.inf]
freq_minmax = [1, 4]

xnt = im.reshape([dt, dx * dy]).T

opt_in_freqdom = True
m = 5
cm = cmm.CMM(
    xnt.astype("float16"),
    m=m,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    freq_minmax=freq_minmax,
    opt_in_freqdom=opt_in_freqdom,
)

cm.optimize(10)

cm.save_results(savepath)

t1 = time()
print(f"time: {int((t1-t0))}")
