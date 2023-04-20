from scipy.sparse.linalg import LinearOperator, eigsh

def compute_first_singular_vector(A):
    """This is taken from scipy.sparse.linalg's implementation of svds"""

    n, m = A.shape
    if n >= m:
        def matmat(X):
            return A.conj().transpose() @ (A @ X)
        def matvec(x):
            return A.conj().transpose() @ (A @ x)

    else:
        def matmat(X):
            return A @ (A.conj().transpose() @ X)
        def matvec(x):
            return A @ (A.conj().transpose() @ x)

    XH_X = LinearOperator(shape=(n, n), dtype=A.dtype,
                          matmat=matmat,
                          matvec=matvec)

    eigval, eigvec = eigsh(XH_X, k=1, tol=0, which='LM')
    return eigval, eigvec
