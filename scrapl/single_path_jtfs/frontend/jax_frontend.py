from ..._dependencies import require_backend

require_backend("jax")

from kymatio.scattering1d.frontend.jax_frontend import TimeFrequencyScatteringJax

from .base_frontend import TimeFrequencyScraplBase


class TimeFrequencyScraplJax(TimeFrequencyScraplBase, TimeFrequencyScatteringJax):
    """Single-path JTFS with native JAX arrays and automatic differentiation.

    Construct the transform outside JIT and keep ``n2`` and ``n_fr`` static
    when compiling ``scattering_singlepath``. Coefficients follow the existing
    SCRAPL padding convention, without PyTorch's trailing real/complex axis.
    """

    def __init__(self, **kwargs):
        kwargs["out_type"] = "array"
        kwargs["format"] = "joint"
        TimeFrequencyScatteringJax.__init__(self, **kwargs)


__all__ = ["TimeFrequencyScraplJax"]
