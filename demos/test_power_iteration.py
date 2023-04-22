import numpy as np
from cmm import power_iteration as pi
from jax import vmap

from scipy.linalg import eigh
from cmm.cmm_funcs import (
    compute_cluster_mean_minimal,
    compute_cluster_mean_minimal_fast,
)
from cmm.utils import timeit
from time import time
from scipy.sparse.linalg import svds
from scipy.linalg import svd

np.random.seed(3)

n = 10
k = 5
f = 8
xnkf = np.random.randn(n, k, f) + 1j * np.random.randn(n, k, f)
xnk = xnkf[:, :, 0]


a = 1 + 1j
b = 1 + 2j
a /= np.linalg.norm(a)
b /= np.linalg.norm(b)

c = a * b
d = a / b
e = a * np.conj(b)
print(c, d, e)
# t0 = time()
# eigvecs, eigvals = compute_cluster_mean_minimal(xnkf, normalize=False)

# timeit(t0)
# t0 = time()
# pi_eigvals, pi_eigvecs = vmap(pi.power_iteration_jit)(xnkf.transpose([2, 0, 1]))
# timeit(t0)
# print(pi_eigvecs.shape)
# A = xnk.T @ xnk.conj()
# ev, evec = eigh(A, subset_by_index=[k - 1, k - 1])

# u1, s1, vh1 = svds(xnkf[:, :, 0], k=1, tol=0)

# u, s, vh = svd(xnkf[:, :, 0], full_matrices=False)
# u1, s1, vh1 = svds(xnkf[:, :, 0], k=1, tol=0, which="LM")
# print(np.allclose(u[:, 0], u1[:, 0]))

# print(u[:, 0] / u1[:, 0])
# print(vh[0] / pi_eigvecs[0])
# print()

# vh = vh.conj(
# print(s[0] ** 2, ev[0], pi_eigvals[0])
# print(np.linalg.norm(vh[0] - evec[:, 0]))
# print(np.linalg.norm(vh[0] - pi_eigvecs[0]))
# print(evec[0, 0], vh[0, 0], pi_eigvecs[0, 0])


# print("\n")
# print(
#     eigvecs.shape,
#     pi_eigvecs.shape,
#     evec.shape,
# )


# print(
#     f" diff in power ite singular value: {np.linalg.norm(pi_eigvals[0] - eigvals[0])}"
# )
# print(f" diff in singular value: {np.linalg.norm(ev[0] - eigvals[0])}")

# print(f" diff in vector  power ite norm: {np.linalg.norm(eigvecs[0] - pi_eigvecs[0])}")
# print(f" diff in vector scipy norm: {np.linalg.norm(eigvecs[0] - evec[:,0])}")
# print("\n\n")
# print(v[0])
# print("\n")
# print(evec[:, 0])
# # # t0 = time()
# # # eigvecs_fast, eigvals_fast = compute_cluster_mean_minimal_fast(xnkf)
# # # print(eigvals_fast, eigvecs_fast.shape)
# # # timeit(t0)

# #


# # # print(s**2, vh.shape, u.shape)
# # print(np.linalg.norm(vh[0] + eigvecs[0]))
# # print(np.linalg.norm(vh[0] * s + eigvecs[0]))
# # print(np.linalg.norm(vh[0] * s**2 + eigvecs[0]))
# # print(
# #     vh[0],
# # )
# # print(-eigvecs[0])
# # print(
# #     vh[0],
# # )
# # print(vh[0] / eigvecs[0])


# def test_power_iteration():
#     # M = xnk.T @ np.conj(xnk)
#     # eigval, eigvec = eigh(M, subset_by_index=[k - 1, k - 1])
#     verbose = False
#     itemax = 1e5
#     myeigval, myeigvec = pi.power_iteration(xnk, itemax=itemax, verbose=verbose)
#     # allclose = np.allclose(myeigval, eigval)
#     # print(allclose)


# # test_power_iteration()
