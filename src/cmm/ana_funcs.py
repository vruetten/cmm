import numpy as np
from cmm import cmm
from cmm import utils


def reload_model(result_path):
    result = np.load(result_path, allow_pickle=True).item()
    nperseg = result["nperseg"]
    fs = result["fs"]
    noverlap = result["noverlap"]
    freq_minmax = result["freq_minmax"]
    dxdy = result["dxdy"]
    m = result["m"]
    data_path = result["data_path"]
    xnt = utils.load_data(data_path=data_path)

    cm = cmm.CMM(
        xnt,
        m=m,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        freq_minmax=freq_minmax,
        dxdy=dxdy,
    )

    return cm
