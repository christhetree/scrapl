from kymatio.frontend.entry import ScatteringEntry
from .torch_frontend import TimeFrequencyScraplTorch


class TimeFrequencyScraplEntry(ScatteringEntry):
    def __init__(self, *args, **kwargs):
        # Check if backend is torch (default to torch if not specified)
        backend = kwargs.get("backend", "torch")

        if backend == "torch":
            # Manually switch the class to your local Torch implementation
            self.__class__ = TimeFrequencyScraplTorch
            # Re-initialize with the new class
            self.__init__(*args, **kwargs)
        else:
            # Fallback to default behavior for other backends (will likely fail
            # if you haven't implemented them, but preserves original logic)
            super().__init__(name="ScRaPL", class_name="scattering1d", *args, **kwargs)


__all__ = ["TimeFrequencyScraplEntry"]
