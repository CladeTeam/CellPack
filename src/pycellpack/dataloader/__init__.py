"""User-facing single-cell datasets, sample access sequences and training batches.

Import and configuration require NumPy and the native extension only. Torch
is loaded when iteration begins; install it separately for your environment.
"""

from ..loader.access_fn import AccessFn
from ..loader.process_fn import OutputLayout, ProcessFn
from ..loader.process_fn import presets as process_presets
from ._accesses import Accesses
from ._dataset import ScDataset
from ._loader import ScDataLoader

__all__ = ["AccessFn", "Accesses", "OutputLayout", "ProcessFn", "ScDataLoader", "ScDataset", "process_presets"]
