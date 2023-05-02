import sys, os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as pl
import colorcet as cc
from cmm import utils
from cmm import cmm
from time import time
from cmm import ana_funcs
from cmm.utils import timeit

sys.path.append("/groups/ahrens/home/ruttenv/code/zfish/")
from zfish.util import filesys as fs


sample = "f337"
sample = "cardio"
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
    results_path = dirs["cmm"] + "cmm_run/"
    save_path = dirs["cmm"] + "cmm_run/plots/"
    os.makedirs(save_path, exist_ok=True)

if sample == "cardio":
    ###` CARDIO DATA
    ms = [2, 5, 10, 20, 30, 50]
    base_dir = "/nrs/ahrens/Virginia_nrs/behavior_rig_flow/230304_f474_9dpf_casper/"
    fnum = fs.get_fnum(base_dir)
    print(f"fnum: {fnum}")
    exp = 0
    dirs_ = fs.get_subfolders(base_dir)
    folder_path = dirs_[f"exp{exp}"]
    dirs = fs.get_subfolders(folder_path)
    results_path = dirs["cmm"] + "cmm_run/"
    save_path = dirs["cmm"] + "cmm_run/plots/"
    os.makedirs(save_path, exist_ok=True)

results_paths = sorted(glob(results_path + "results_nperseg*.npy"))
optimization_time = []
ms = []
silhouettes = []


for ind, path in enumerate(results_paths):
    res = np.load(path, allow_pickle=True).item()
    m = res["m"]
    ms.append(m)
    labels_im = res["labels_im"]
    optimization_time.append(res["optimization_time"])
    labels = res["labels"]
    freqs = res["freqs"]
    coherence_mn = res["coherence_mnf"].mean(-1)
    cluster_coherence_m2f = res["cluster_coherence_m2f"]

    if hasattr(res, "labels_im_valid"):
        labels_im = res["labels_im_valid"]
    else:
        threshold = 0.7
        valid_clusters, labels_im = cmm.threshold_clusters(
            labels_im=labels_im,
            cluster_coherence_m2f=cluster_coherence_m2f,
            threshold=threshold,
        )

    silhouette = utils.compute_silhouette_proxy(
        coherence_mn=coherence_mn, labels=labels
    )
    silhouettes.append(silhouette)
    cluster_coherence_m2 = res["cluster_coherence_m2f"].mean(-1)

    ### PLOT LABELS
    fig = pl.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(labels_im, cmap=cc.cm.glasbey, interpolation="nearest")
    pl.title(f"m: {m}")
    pl.savefig(save_path + f"labels_m{m}", bbox_inches="tight", transparent=True)

    ### GRAPH CLUSTER COHERENCE
    fig = pl.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    order = np.argsort(cluster_coherence_m2[:, 0]).squeeze()[::-1]
    ax.plot(cluster_coherence_m2[order, 0], "o-")
    ax.plot(cluster_coherence_m2[order, 1], "o-")
    ax.set_ylim([0, 1])
    pl.title(f"m: {m}")
    pl.savefig(
        save_path + f"cluster_coherence_m{m}", bbox_inches="tight", transparent=True
    )
    ### GRAPH COHERENCE AS FUNCTION OF FREQUENCY
    cluster_order = np.argsort(np.max(cluster_coherence_m2f[:, 0], axis=-1))[::-1]
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.set_title("cluster coherence")
    im = ax.pcolormesh(freqs, np.arange(m), cluster_coherence_m2f[cluster_order, 0])
    ax.set_xlabel("frequency")
    ax.set_ylabel("cluster #")
    fig.colorbar(im, fraction=0.04)

    pl.savefig(
        save_path + f"cluster_coherence_mat_m{m}", bbox_inches="tight", transparent=True
    )

    ### GRAPH COHERENCE AS FUNCTION OF FREQUENCY


### GRAPH SILHOUETTE
fig = pl.figure(figsize=(10, 3))
ax = fig.add_subplot(111)
ax.plot(ms, silhouettes, "o-")
ax.set_xlabel("cluster #")
ax.set_ylabel("silhouette")
pl.title(f"silhouette")
pl.savefig(save_path + f"silhouette", bbox_inches="tight", transparent=True)

### PLOT RUN TIME
fig = pl.figure(figsize=(10, 3))
ax = fig.add_subplot(111)
ax.plot(ms, optimization_time, "o-")
ax.set_xlabel("cluster #")
ax.set_ylabel("run time")
pl.title(f"optimization time")
pl.savefig(save_path + f"optimization_time", bbox_inches="tight", transparent=True)
