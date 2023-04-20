import numpy as np
from scipy.linalg import eigh
from cmm import power_iteration as pi
from cmm.cmm_funcs import (
    compute_cluster_mean_minimal,
    compute_cluster_mean_minimal_fast,
)
from cmm.utils import timeit
from time import time


np.random.seed(0)

n = 20
k = 10
f = 3
xnkf = np.random.randn(n, k, f) + 1j * np.random.randn(n, k, f)
xnk = xnkf[:, :, 0]


def test_power_iteration():
    M = xnk.T @ np.conj(xnk)
    eigval, eigvec = eigh(M, subset_by_index=[k - 1, k - 1])
    verbose = False
    itemax = 1e5
    myeigval, myeigvec = pi.power_iteration(xnk, itemax=itemax, verbose=verbose)
    allclose = np.allclose(myeigval, eigval)
    print(allclose)


t0 = time()
eigvecs, eigvals = compute_cluster_mean_minimal(xnkf)
print(eigvals, eigvecs.shape)
timeit(t0)

t0 = time()
eigvecs_fast, eigvals_fast = compute_cluster_mean_minimal_fast(xnkf)
print(eigvals_fast, eigvecs_fast.shape)
timeit(t0)
