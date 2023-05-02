import argparse
import pandas as pd
import numpy as np
from cmm import cmm
from cmm import utils
import re
from time import time

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", "-p", type=str, help="path to data file")
parser.add_argument("--save_path", "-sp", type=str, help="path to save results")
parser.add_argument("--cluster_num", "-m", type=int, help="cluster number")
parser.add_argument("--nperseg", "-nperseg", type=int, help="nperseg")
parser.add_argument(
    "--freq_range", "-freq", nargs="+", help="frequency range", default=None
)
parser.add_argument("--sampling_freq", "-fs", type=float, help="sampling rate")
parser.add_argument("--dxdy", "-dxdy", nargs="+", help="xy dimension", default=None)

args = parser.parse_args()


data_path = args.data_path
save_path = args.save_path

m = args.cluster_num
fs = args.sampling_freq
nperseg = args.nperseg
noverlap = int(0.7 * nperseg)
freq_minmax = [float(i) for i in args.freq_range]
dxdy = [int(i) for i in args.dxdy]
print(f"freq minmax:{freq_minmax}")
t0 = time()
xnt = utils.load_data(data_path=data_path)

print("data loaded")

t1 = time()
tt = int(t1 - t0)
print(f"time: {tt}")

t0 = time()
cm = cmm.CMM(
    xnt,
    m=m,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    freq_minmax=freq_minmax,
    dxdy=dxdy,
    nprocesses=2,
)

cm.optimize(200)
t1 = time()
tt = int(t1 - t0)
print(f"time: {tt}")
print("starting to save results")
cm.analyse_results()
results = cm.store_results()
results["data_path"] = data_path
results["save_path"] = save_path
# silhouette = cm.compute_model_silhouette() # very expensive to compute
# results['silhoette'] = silhouette
savepath = (
    save_path
    + f"results_nperseg_{nperseg}_m{str(m).zfill(3)}_freq{freq_minmax[0]}_{freq_minmax[1]}"
)
savepath = re.sub(r"[.]", r"p", savepath)
cm.save_results(results, savepath)
