import numpy as np
from cmm import power_iteration as pi
from jax import vmap

# from scipy.linalg import eigh
from cmm.cmm_funcs import (
    compute_cluster_mean_minimal,
    compute_cluster_mean_minimal_fast,
)
from cmm.utils import timeit
from time import time
from scipy.linalg import eigh


np.random.seed(3)

n = 3000
k = 100
f = 20
xnkf = np.random.randn(n, k, f) + 1j * np.random.randn(n, k, f)
xnk = xnkf[:, :, 0]

t0 = time()
eigvecs, eigvals = compute_cluster_mean_minimal(xnkf, normalize=False)

timeit(t0)
t0 = time()
pi_eigvals, pi_eigvecs = vmap(pi.power_iteration_jit)(xnkf.transpose([2, 0, 1]))
timeit(t0)

A = xnk.T @ np.conj(xnk)
ev, evec = eigh(A, subset_by_index=[k - 1, k - 1])

print("\n")
print(
    eigvecs.shape,
    pi_eigvecs.shape,
    evec.shape,
)


print(
    f" diff in power ite singular value: {np.linalg.norm(pi_eigvals[0] - eigvals[0])}"
)
print(f" diff in singular value: {np.linalg.norm(ev[0] - eigvals[0])}")

print(f" diff in vector  power ite norm: {np.linalg.norm(eigvecs[0] - pi_eigvecs[0])}")
print(f" diff in vector scipy norm: {np.linalg.norm(eigvecs[0] - evec[:,0])}")
print("\n\n")
print(ev[0])
print("\n")
print(evec[:, 0])
# # t0 = time()
# # eigvecs_fast, eigvals_fast = compute_cluster_mean_minimal_fast(xnkf)
# # print(eigvals_fast, eigvecs_fast.shape)
# # timeit(t0)

# from scipy.sparse.linalg import svds

# u, s, vh = svds(np.conj(xnkf[:, :, 0]), k=1, tol=0)

# # print(s**2, vh.shape, u.shape)
# print(np.linalg.norm(vh[0] + eigvecs[0]))
# print(np.linalg.norm(vh[0] * s + eigvecs[0]))
# print(np.linalg.norm(vh[0] * s**2 + eigvecs[0]))
# print(
#     vh[0],
# )
# print(-eigvecs[0])
# print(
#     vh[0],
# )
# print(vh[0] / eigvecs[0])


def test_power_iteration():
    # M = xnk.T @ np.conj(xnk)
    # eigval, eigvec = eigh(M, subset_by_index=[k - 1, k - 1])
    verbose = False
    itemax = 1e5
    myeigval, myeigvec = pi.power_iteration(xnk, itemax=itemax, verbose=verbose)
    # allclose = np.allclose(myeigval, eigval)
    # print(allclose)


# test_power_iteration()
