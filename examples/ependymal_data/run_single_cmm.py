import numpy as np
from cmm import cmm
import sys, os

sys.path.append("/groups/ahrens/home/ruttenv/code/zfish/")
from zfish.util import filesys as fs
import pandas as pd
from time import time
import re


### LOAD DATA
base_dir = "/nrs/ahrens/Virginia_nrs/confocal_nikon_wbi/221123_f337_ubi_gCaMP7f_8506_8dpf_hypoxia_tricaine_oscillation/"
fnum = fs.get_fnum(base_dir)
print(f"fnum: {fnum}")

exp = 0
dirs_ = fs.get_subfolders(base_dir)
folder_path = dirs_[f"exp{exp}"]
dirs = fs.get_subfolders(folder_path)
os.makedirs(dirs["main"] + "cmm/ron/", exist_ok=True)
dirs = fs.get_subfolders(folder_path)
plotpath = dirs["cmm"] + "ron/"

df = pd.read_hdf(dirs["ephys"] + "xnt_denoised.h5", key="data")
x, y = 96, 426
t = 1200
xnt = np.array(df.T)[:, :t]
print(xnt.shape)

### SET PARAMETERS
fs = 0.3
nperseg = int(fs * 1200)
noverlap = int(0.7 * nperseg)
print("parameters set")
## RUN CMM
m = 3
freq_minmax = [0, 0.005]
t0 = time()
cm = cmm.CMM(
    xnt,
    m=m,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    freq_minmax=freq_minmax,
    dxdy=[x, y],
)

cm.optimize(200)
t1 = time()
tt = int(t1 - t0)
print(f"time: {tt}")
print("starting to save results")

results = cm.analyse_results()
cm.store_results()
savepath = (
    dirs["cmm"]
    + f"results_nperseg_{nperseg}_m{m}_freq{freq_minmax[0]}_{freq_minmax[1]}"
)
savepath = re.sub(r"[.]", r"p", savepath)
cm.save_results(savepath)
