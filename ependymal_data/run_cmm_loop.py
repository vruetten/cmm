import sys, os
import pandas as pd
import numpy as np

sys.path.append("/groups/ahrens/home/ruttenv/code/zfish/")
from zfish.util import filesys as fs
from glob import glob
import tifffile as tf
import zarr

sample = "f337"
# sample = "cardio"
# sample = "f338"

if sample == "f337":
    ### F337 EPENDYMAL CELL DATA
    base_dir = "/nrs/ahrens/Virginia_nrs/confocal_nikon_wbi/221123_f337_ubi_gCaMP7f_8506_8dpf_hypoxia_tricaine_oscillation/"
    fnum = fs.get_fnum(base_dir)
    print(f"fnum: {fnum}")
    exp = 0
    dirs_ = fs.get_subfolders(base_dir)
    folder_path = dirs_[f"exp{exp}"]
    dirs = fs.get_subfolders(folder_path)
    save_path = dirs["cmm"] + "cmm_run/"
    os.makedirs(save_path, exist_ok=True)

    data_path = dirs["ephys"] + "xnt_denoised_cropped.h5"
    x, y = 96, 426
    dxdy = [x, y]
    fps = 0.3
    nperseg = int(fps * 1200)
    freq_minmax = [0, 0.002]
    ms = [2, 5, 10, 20, 30, 50, 80]

if sample == "cardio":
    ###` CARDIO DATA
    ms = [
        2,
        5,
        10,
        20,
        30,
        50,
    ]
    base_dir = "/nrs/ahrens/Virginia_nrs/behavior_rig_flow/230304_f474_9dpf_casper/"
    fnum = fs.get_fnum(base_dir)
    print(f"fnum: {fnum}")
    exp = 0
    dirs_ = fs.get_subfolders(base_dir)
    folder_path = dirs_[f"exp{exp}"]
    dirs = fs.get_subfolders(folder_path)
    os.makedirs(dirs["main"] + "cmm", exist_ok=True)
    save_path = dirs["cmm"] + "cmm_run/"
    os.makedirs(save_path, exist_ok=True)

    # impath = glob(dirs["imag_crop"] + "*.tif")[0]
    # imzarr = tf.imread(impath, aszarr=True)
    # im = zarr.open(imzarr, mode="r")[: 15 * 200]
    # t, dx, dy = im.shape
    # xnt = im.reshape([t, dx * dy]).T
    t = str(15 * 200)
    data_path = dirs["ephys"] + f"xnt_{t}.npy"
    # np.save(data_path, xnt)
    # print(dx, dy)
    x, y = 80, 70
    dxdy = [x, y]

    freq_minmax = [1.5, 3]
    fps = 15
    nperseg = int(fps * 20)

if sample == "f338":
    ## F338 DATA
    ms = [5, 10, 30, 50, 100]
    base_dir = "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia/exp0/"
    dirs = fs.get_subfolders(base_dir)
    data_path = dirs["ephys"] + "xnt_denoised.h5"
    fps = 0.267
    nperseg = int(fps * 600)
    dxdy = [0, 0]
    freq_minmax = [0, 0]


for m in ms:
    SCRIPTPATH = (
        "/groups/ahrens/home/ruttenv/python_packages/cmm/ependymal_data/run_cmm_cmd.py"
    )

    cmd = f"python {SCRIPTPATH} -p {data_path} -sp  {save_path} -m  {m} -nperseg {nperseg} -freq {freq_minmax[0]} {freq_minmax[1]} -fs {fps} -dxdy {dxdy[0]} {dxdy[1]}"

    os.system(cmd)
    print(cmd)


### F337 EPENDYMAL CELL DATA
# t = 1200
# df = pd.read_hdf(dirs["ephys"] + "xnt_denoised.h5", key="data")[:t]
# df = pd.read_hdf(dirs["ephys"] + "xnt_denoised_cropped.h5", key="data")
# df.to_hdf(data_path, key="data")
