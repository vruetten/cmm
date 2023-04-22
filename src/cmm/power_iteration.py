import jax.numpy as jnp
from jax import jit
import jax
from jax.lax import while_loop


def inner_product(val):
    Ank, vk, diff = val
    k = Ank.shape[1]
    Av = Ank.conj() @ vk
    AAv = Ank.T @ Av
    vk_new = AAv / jnp.linalg.norm(AAv)
    diff = jnp.linalg.norm(vk_new - vk)
    return Ank, vk_new, diff


def cond_fun(val):
    Ank, vk, diff = val
    return diff > 0.001


def power_iteration(Ank: jnp.array):
    """power iteration for Ank"""
    n, k = Ank.shape
    vk = jnp.arange(k) + 1j * jnp.arange(k)
    vk = vk / jnp.linalg.norm(vk)
    init_val = (Ank, vk, 1)
    val = while_loop(cond_fun, inner_product, init_val)
    vk = val[1]
    Av = Ank.conj() @ vk
    eigenvalue = (Av * Av.conj()).sum().real
    return eigenvalue, vk


power_iteration_jit = jax.jit(power_iteration)
