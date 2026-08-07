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
sample = "f338"

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
    threshold = 0.7

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
    threshold = 0.7

if sample == "f338":
    ## F338 DATA
    base_dir = "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia/exp0/"
    results_path = dirs["cmm"] + "cmm_run/"
    save_path = dirs["cmm"] + "cmm_run/plots/"
    threshold = 0.7

results_paths = sorted(glob(results_path + "results_nperseg*.npy"))
optimization_time = []
ms = []
silhouettes = []
silhouettes_proxy = []

for ind, path in enumerate(results_paths):
    res = np.load(path, allow_pickle=True).item()
    im = res["im"]
    m = res["m"]
    ms.append(m)
    optimization_time.append(res["optimization_time"])
    silhouettes.append(res["silhouette"])
    silhouettes_proxy.append(res["silhouette_proxy"])

    labels = res["labels"]
    freqs = res["freqs"]
    coherence_mn = res["coherence_mnf"].mean(-1)
    cluster_coherence_m2f = res["cluster_coherence_m2f"]
    cluster_coherence_m2 = cluster_coherence_m2f.mean(-1)
    ymt = res["ymt"]
    ymt_coh = res["ymt_coh"]
    freqs = res["freqs"]
    if im:
        dx, dy = res["dxdy"]
        labels_im = res["labels_im"]

    if hasattr(res, "labels_im_valid"):
        labels_im_valid = res["labels_im_valid"]
    else:
        valid_clusters, labels_im_valid = cmm.threshold_clusters(
            labels_im=labels_im,
            cluster_coherence_m2f=cluster_coherence_m2f,
            threshold=threshold,
        )

    angles_mf_im_mean = res["angles_mf_im_mean"]
    coherence_mf_im = (
        res["coherence_mnf"].reshape([m, dx, dy, -1]).transpose([0, 3, 1, 2])
    )

    ### PLOT LABELS
    if im:
        fig = pl.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        # ax.imshow(labels_im, cmap=cc.cm.glasbey, interpolation="nearest")
        ax.imshow(labels_im, cmap="tab20", interpolation="nearest")
        pl.title(f"clusters - m: {m}")
        pl.savefig(save_path + f"labels_m{m}", bbox_inches="tight", transparent=True)

        fig = pl.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        # ax.imshow(labels_im, cmap=cc.cm.glasbey, interpolation="nearest")
        ax.imshow(labels_im_valid, cmap="tab20", interpolation="nearest")
        pl.title(f"clusters - m: {m}")
        pl.savefig(
            save_path + f"labels_valid_m{m}", bbox_inches="tight", transparent=True
        )

    ### GRAPH CLUSTER COHERENCE
    fig = pl.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    cluster_order = np.argsort(cluster_coherence_m2[:, 0]).squeeze()[::-1]
    ax.plot(
        cluster_coherence_m2f[cluster_order, 0].max(-1),
        "o-",
        label="max inner clus. coh.",
    )
    ax.plot(
        cluster_coherence_m2f[cluster_order, 1].max(-1),
        "o-",
        label="max outer clus. coh.",
    )
    ax.plot(cluster_coherence_m2[cluster_order, 0], ".-", label="mean inner clus. coh.")
    ax.plot(cluster_coherence_m2[cluster_order, 1], ".-", label="mean outer clus. coh.")
    ax.set_ylim([0, 1])
    ax.set_xlabel("cluster #")
    ax.set_ylabel("coherence")
    pl.legend()
    pl.title(f"m: {m}")
    pl.savefig(
        save_path + f"cluster_coherence_m{m}", bbox_inches="tight", transparent=True
    )
    ### GRAPH COHERENCE AS FUNCTION OF FREQUENCY
    cluster_metric = np.max(cluster_coherence_m2f[:, 0], axis=-1) - np.max(
        cluster_coherence_m2f[:, 1], axis=-1
    )
    cluster_order = np.argsort(cluster_metric)[::-1]
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.set_title("cluster coherence")
    im = ax.pcolormesh(
        freqs, np.arange(m), cluster_coherence_m2f[cluster_order, 0], vmin=0, vmax=1
    )
    ax.set_xlabel("frequency")
    ax.set_ylabel("cluster #")
    fig.colorbar(im, fraction=0.04)

    pl.savefig(
        save_path + f"cluster_coherence_mat_m{m}", bbox_inches="tight", transparent=True
    )

    ### GRAPH COHERENCE MAP, PHASE, LATENT
    if im:
        for ind, cl in enumerate(cluster_order):
            coh = np.max(cluster_coherence_m2f[cl, 0])
            if coh > 0.8:
                fig, axs = pl.subplots(nrows=2, ncols=3, figsize=(21, 8))
                argfreq = np.argmax(cluster_coherence_m2f[cl, 0])
                alpha = coherence_mf_im[cl].max(0)
                im00 = axs[0, 0].imshow(alpha, vmin=0, vmax=1, cmap="coolwarm")
                axs[0, 0].set_title("coherence")
                im01 = axs[0, 1].imshow(
                    angles_mf_im_mean[cl, argfreq], cmap="seismic", alpha=alpha
                )
                freq = np.round(freqs[argfreq], 2)
                axs[0, 1].set_title(f"phase (freq: {freq}Hz)")
                tmp = np.zeros_like(labels_im)
                tmp[labels_im == cl] = 1
                axs[0, 2].imshow(tmp, cmap="seismic")
                axs[0, 2].set_title("mask")
                axs[1, 0].plot(ymt[cl, 200:500])
                axs[1, 1].plot(ymt_coh[cl, 200:500])
                fig.colorbar(
                    im00,
                )
                fig.colorbar(
                    im01,
                )
                pl.savefig(
                    save_path + f"coherence_phase_m{m}_top{ind}_latent_{cl}",
                    bbox_inches="tight",
                    transparent=True,
                )


### GRAPH SILHOUETTE
fig = pl.figure(figsize=(10, 3))
ax = fig.add_subplot(111)
ax.plot(ms, silhouettes, "o-")
ax.set_xlabel("cluster #")
ax.set_ylabel("silhouette")
pl.title(f"silhouette")
pl.savefig(save_path + f"silhouette", bbox_inches="tight", transparent=True)

fig = pl.figure(figsize=(10, 3))
ax = fig.add_subplot(111)
ax.plot(ms, silhouettes_proxy, "o-")
ax.set_xlabel("cluster #")
ax.set_ylabel("silhouette proxy")
pl.title(f"silhouette")
pl.savefig(save_path + f"silhouette_proxy", bbox_inches="tight", transparent=True)


### PLOT RUN TIME
fig = pl.figure(figsize=(10, 3))
ax = fig.add_subplot(111)
ax.plot(ms, optimization_time, "o-")
ax.set_xlabel("cluster #")
ax.set_ylabel("run time")
pl.title(f"optimization time")
pl.savefig(save_path + f"optimization_time", bbox_inches="tight", transparent=True)
