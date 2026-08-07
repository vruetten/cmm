def compute_cluster_mean_minimal_scan(
    xnt: np.array,
):
    # print('compiling')
    xnkf_coefs = xnt
    n, k, f = xnkf_coefs.shape
    t0 = time()
    pn_f = np.sqrt(np.einsum("nkf, nkf->nf", xnkf_coefs, np.conj(xnkf_coefs)))
    xnkf_coefs_normalized = xnkf_coefs / pn_f[:, None]

    # init = np.zeros([k, k, f]).astype("complex64")
    # pkkf, _ = scan(fold, init, xnkf_coefs_normalized)

    pkkf = np.einsum(
        "nkf, nlf->klf", xnkf_coefs_normalized, np.conj(xnkf_coefs_normalized)
    )  # if n filled with zeroes not a problem
    t1 = time()
    Vp = [scieigh(m, subset_by_index=[k - 1, k - 1]) for m in pkkf.transpose([2, 0, 1])]
    # Vp = [eigh(m) for m in pkkf.transpose([2, 0, 1])]
    t2 = time()

    t11 = np.round((t1 - t0) * 100, 4)
    t22 = np.round((t2 - t1) * 100, 4)
    print(t11, t22)
    eigvals_p = np.array(list(zip(*Vp))[0]).squeeze()
    # [:,-1]
    eigvecs_p_fk = np.array(list(zip(*Vp))[1]).squeeze()
    # [:,:,-1]
    return eigvecs_p_fk, eigvals_p


def fold(carry, x0):
    k, f = x0.shape
    tmp0 = broadcast(x0, sizes=(k,))
    tmp1 = broadcast(x0, sizes=(k,)).transpose(1, 0, 2)
    carry += tmp0 * np.conj(tmp1)  # nx x ny x w
    print(k, f, carry.shape)
    return carry, 0
