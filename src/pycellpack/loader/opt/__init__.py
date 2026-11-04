"""Cache plans, planning options, cost models and the serial page reference.

`Plan` is a compiled sequence of page and row residencies (the `CPKPLAN1` v1
form), `PlanOptions` says what to compile for, and `CostModel` prices the work
of a window so the compiler can rank decisions. `PageTrace` gives the serial
page optimum a plan is read against; `PlanSummary`, `OnlineStats`,
`CacheSummary` and `Prices` are the native compilers' frozen statistics.
Plans and traces own native data and translate it, while options and models
are Python values translated when used; schedules are decided in Rust.
"""

from ..._cellpack import access_fingerprint
from ._config import (
    CONVERSIONS,
    CostModel,
    InputCost,
    OutputCost,
    PlanOptions,
    ProcessCost,
    RowCost,
)
from ._plan import PageTrace, Plan
from ._stats import CacheSummary, OnlineStats, PlanSummary, Prices

__all__ = [
    "CONVERSIONS",
    "CacheSummary",
    "CostModel",
    "InputCost",
    "OnlineStats",
    "OutputCost",
    "PageTrace",
    "Plan",
    "PlanOptions",
    "PlanSummary",
    "Prices",
    "ProcessCost",
    "RowCost",
    "access_fingerprint",
]
