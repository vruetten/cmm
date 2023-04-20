import jax.numpy as np


def power_iteration(Ank: np.array, itemax=1000, verbose=False):
    """power iteration for Akn"""
    n, k = Ank.shape
    vk = np.ones(k) / np.sqrt(k)
    Akn = Ank.T
    Ank_ = np.conj(Ank)
    for ite in range(int(itemax)):
        Av = Ank_ @ vk
        AAv = Akn @ Av
        vk_new = AAv / np.linalg.norm(AAv)

        diff = np.abs(vk_new - vk).sum() / k
        if verbose:
            if diff < 0.001:
                print(f"reached diff limit: {diff}")
                break
            if np.mod(ite, 500) == 0:
                print(f"at ite: {ite}, diff: {diff}")

            if ite == itemax:
                print(f"reached iteration limit - diff{diff} - has not converged!")
        vk = vk_new

    Av = Ank_ @ vk
    eigenvalue = (Av * np.conj(Av)).sum()
    return eigenvalue, vk
