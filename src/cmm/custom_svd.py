from scipy.sparse.linalg import LinearOperator, eigsh


def compute_first_singular_vector(A):
    """This is taken from scipy.sparse.linalg's implementation of svds"""

    n, m = A.shape
    if n >= m:
        XH_X = LinearOperator(shape=(n, n), dtype=A.dtype,
                              matmat=lambda X: A.conj().transpose() @ (A @ X),
                              matvec=lambda x: A.conj().transpose() @ (A @ x))
    else:
        XH_X = LinearOperator(shape=(n, n), dtype=A.dtype,
                              matmat=lambda X: A @ (A.conj().transpose() @ X),
                              matvec=lambda x: A @ (A.conj().transpose() @ x))

    eigval, eigvec = eigsh(XH_X, k=1, tol=0, which='LM')
    return eigval, eigvec
