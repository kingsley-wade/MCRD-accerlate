"""Backend/device capability reporting and adapter boundaries."""

from __future__ import annotations

from .config import Backend


JAX_BACKENDS = (Backend.JAX_CPU, Backend.JAX_METAL, Backend.JAX_CUDA)


def get_jax_device(backend: Backend):
    """Return the concrete JAX device requested by a backend label.

    Explicit routing is important on multi-platform machines: ``jax-cpu`` must
    remain a CPU reference path even when CUDA is installed, while ``jax-cuda``
    must fail rather than silently falling back to CPU.
    """
    if backend not in JAX_BACKENDS:
        raise ValueError(f"backend {backend.value!r} is not a JAX backend")

    import jax

    platform = {
        Backend.JAX_CPU: "cpu",
        Backend.JAX_CUDA: "gpu",
        Backend.JAX_METAL: "METAL",
    }[backend]
    try:
        devices = jax.devices(platform)
    except RuntimeError as exc:
        raise RuntimeError(
            f"requested backend {backend.value!r}, but JAX has no {platform!r} device"
        ) from exc
    if not devices:
        raise RuntimeError(
            f"requested backend {backend.value!r}, but JAX has no {platform!r} device"
        )
    return devices[0]
