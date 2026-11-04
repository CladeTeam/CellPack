"""Native loading of `.cpk` matrices: one data source (each matrix mapping
its columns to standard ids), named output schemas (built-in dense rows or
native process functions, each with its own output dtype), access sequences
(stored or computed), cache plans compiled offline or online, and a
single-host distributed producer.

Subpackages: :mod:`.opt` (cache plans, planning options and cost models),
:mod:`.costs` (measured cost records, resolution and calibration) and
:mod:`.process_fn` (native process functions, their C ABI header and the
model templates under its ``include/template/``)."""

from .._cellpack import (
    MAX_SCHEMAS,
    PROCESS_ABI_VERSION,
    PROCESS_BOUNDS,
    Accesses,
    build_info,
    kernel_set,
)
from . import access_fn, costs, opt, process_fn
from ._config import IoBackend, LoaderConfig, profile_groups, profile_sites
from ._distributed import DistributedLoader, DistributedRun
from ._loader import Loader, Run, load
from ._ring import Batch, Consumer, Ring, RingLayout
from ._schema import OutputSchema, allowed
from ._shared import SharedRing
from ._source import DataSource
from ._stats import (
    IoStats,
    MemoryEstimate,
    OutputStats,
    ProfileReport,
    RunStats,
    SiteReport,
)
from .access_fn import AccessFn
from .opt import (
    CONVERSIONS,
    CostModel,
    InputCost,
    OnlineStats,
    OutputCost,
    PageTrace,
    Plan,
    PlanOptions,
    PlanSummary,
    ProcessCost,
    RowCost,
    access_fingerprint,
)
from .process_fn import (
    Compiler,
    CompilerNotFoundError,
    ProcessBuildError,
    ProcessFn,
    ProcessLibraryError,
    compile_library,
    find_compiler,
)

__all__ = [
    "CONVERSIONS",
    "MAX_SCHEMAS",
    "PROCESS_ABI_VERSION",
    "PROCESS_BOUNDS",
    "AccessFn",
    "Accesses",
    "Batch",
    "Compiler",
    "CompilerNotFoundError",
    "Consumer",
    "CostModel",
    "DataSource",
    "DistributedLoader",
    "DistributedRun",
    "InputCost",
    "IoBackend",
    "IoStats",
    "Loader",
    "LoaderConfig",
    "MemoryEstimate",
    "OnlineStats",
    "OutputCost",
    "OutputSchema",
    "OutputStats",
    "PageTrace",
    "Plan",
    "PlanOptions",
    "PlanSummary",
    "ProcessBuildError",
    "ProcessCost",
    "ProcessFn",
    "ProcessLibraryError",
    "ProfileReport",
    "Ring",
    "RingLayout",
    "RowCost",
    "Run",
    "RunStats",
    "SharedRing",
    "SiteReport",
    "access_fingerprint",
    "access_fn",
    "allowed",
    "build_info",
    "compile_library",
    "costs",
    "find_compiler",
    "kernel_set",
    "load",
    "opt",
    "process_fn",
    "profile_groups",
    "profile_sites",
]
