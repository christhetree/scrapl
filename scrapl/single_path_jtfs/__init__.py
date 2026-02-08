import os
import sys

_submodule_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../kymatio")),
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../pytorch_hessian_eigenthings")
    ),
]
for _submodule_path in _submodule_paths:
    sys.path.append(_submodule_path)

# Make sure that DeprecationWarning within this package always gets printed
### Snippet copied from sklearn.__init__
import re
import warnings

warnings.filterwarnings(
    "always", category=DeprecationWarning, module=r"^{0}.*".format(re.escape(__name__))
)
warnings.filterwarnings(
    "always",
    category=PendingDeprecationWarning,
    module=r"^{0}.*".format(re.escape(__name__)),
)
### End Snippet

__all__ = ["TimeFrequencyScrapl"]

from .frontend.entry import TimeFrequencyScraplEntry as TimeFrequencyScrapl
