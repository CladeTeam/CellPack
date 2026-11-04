"""Reusable scans components over NumPy buffers; algorithms execute in Rust.

Every allocating function accepts out= where applicable. Workspaces use uint64
NumPy arrays and can be reused between independent encode/decode calls.
Record-level functions take record bytes, a :class:`PagedRecord` or a
``cell=`` of the metadata's own buffer.
"""

from ... import _cellpack
from ._codec import (
    ChunkSizes,
    chunk_sizes,
    decode_chunk,
    decode_csr,
    decode_dense,
    decode_parts,
    decode_record,
    decode_record_dense,
    decode_workspace_bound,
    dense_bytes,
    encode_chunk,
    encode_record,
    encode_workspace_words,
    pack_chunk,
    packed_len,
    record_bound,
    record_nnz,
    selection_nnz,
)
from ._mapping import cast_values, compose_mapping, map_dense, map_parts, map_parts_at
from ._prepared import Decoder, PreparedEncoder
from ._types import (
    DEFAULT_LANE_MIN,
    DEFAULT_MAX_LANES,
    DEFAULT_MODE_SHARE,
    DEFAULT_MU_INDICES,
    DEFAULT_MU_VALUES,
    UNMAPPED,
    DecodeTables,
    Dict,
    EncodeOptions,
    EncodeTables,
    Fill,
    IdSpace,
    LayoutChoice,
    Meta,
    ModeDict,
    PagedRecord,
    Parts,
    Ranks,
    RecordLayout,
    SparseRow,
    Thresholds,
    ValueChoice,
    Values,
)


def kernel_set() -> str:
    """The kernel set bound at import."""
    return _cellpack.kernel_set()


def supported_kernel_sets() -> list[str]:
    """Kernel sets this machine supports."""
    return _cellpack.scans_kernel_sets()


__all__ = [
    "DEFAULT_LANE_MIN",
    "DEFAULT_MAX_LANES",
    "DEFAULT_MODE_SHARE",
    "DEFAULT_MU_INDICES",
    "DEFAULT_MU_VALUES",
    "UNMAPPED",
    "ChunkSizes",
    "DecodeTables",
    "Decoder",
    "Dict",
    "EncodeOptions",
    "EncodeTables",
    "Fill",
    "IdSpace",
    "LayoutChoice",
    "Meta",
    "ModeDict",
    "PagedRecord",
    "Parts",
    "PreparedEncoder",
    "Ranks",
    "RecordLayout",
    "SparseRow",
    "Thresholds",
    "ValueChoice",
    "Values",
    "cast_values",
    "chunk_sizes",
    "compose_mapping",
    "decode_chunk",
    "decode_csr",
    "decode_dense",
    "decode_parts",
    "decode_record",
    "decode_record_dense",
    "decode_workspace_bound",
    "dense_bytes",
    "encode_chunk",
    "encode_record",
    "encode_workspace_words",
    "kernel_set",
    "map_dense",
    "map_parts",
    "map_parts_at",
    "pack_chunk",
    "packed_len",
    "record_bound",
    "record_nnz",
    "selection_nnz",
    "supported_kernel_sets",
]
