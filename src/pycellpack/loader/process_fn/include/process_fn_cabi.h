#ifndef PYCELLPACK_PROCESS_FN_CABI_H
#define PYCELLPACK_PROCESS_FN_CABI_H
#include <stdint.h>
#define PYCELLPACK_PROCESS_ABI_VERSION 1
#define PYCELLPACK_PROCESS_CAPSULE_NAME "pycellpack.process.v1"

/* A schema's process functions: one cell's row, mapped to the schema's
 * output dims and converted to its dtype, in; the cell's fixed-size output
 * (cell_bytes at `out`) out. Providers retain code, vtable and context
 * through their capsule/owner. Functions are native and thread-safe, write
 * only their cell's output and their scratch (nothing is checked), retain
 * no borrowed pointers, and return nonzero to stop the run with that code.
 *
 * Row semantics: an output dim whose gene the cell's matrix has is present
 * (the stored value, or 0 where the record stores none: CSR is built by
 * dropping zeros, so they are expression values); an output dim the matrix
 * has no column for is missing; a matrix column no output dim takes is
 * dropped. CSR rows list stored elements only; dense rows hold 0 at present
 * zeros and the schema default at missing dims.
 *
 * The ProcessFn's row_format picks the rows delivered. "auto" needs both
 * functions: CSR records arrive as CSR rows; dense records use internal
 * dimension, conversion, value-coding and cache thresholds. Cached dense
 * records may compact to CSR at admission. "csr" / "dense" deliver that format and need
 * only that function; the other vtable entry may be NULL. */

/* dtype codes: 0 u8, 1 u16, 2 u32, 3 u64, 4 i8, 5 i16, 6 i32, 7 i64, 8 f32, 9 f64. */
typedef struct {
    uint32_t abi_version, struct_size;
    uint64_t position, cell; /* position in the sequence; global cell id */
    uint32_t schema, dtype;
    const void *user_data;
} PyCellPackContext;

/* Resolves partial rows: positions address stored elements (NULL with
 * count == nnz: every element in order); values are the schema's dtype.
 * Valid only during its call. */
typedef int32_t (*PyCellPackResolve)(const void *row, const uint32_t *positions, uint64_t count, uint32_t *columns, void *values);

/* CSR row: stored elements, output columns in the chunk's coded order (not
 * sorted). Other output dims are zeros or missing; the row cannot tell
 * which. Partial rows have NULL columns/values and must use the resolver. */
typedef struct {
    PyCellPackContext context;
    uint64_t out_dim, nnz;
    const uint32_t *columns;
    const void *values;
    const void *row;
    PyCellPackResolve resolve;
    uint8_t *out;
    uint8_t *scratch;
    uint64_t scratch_bytes;
} PyCellPackCsrRow;

/* Dense row: out_dim values of the schema's dtype: stored values, 0 at
 * present dims without one, the schema default at missing dims. */
typedef struct {
    PyCellPackContext context;
    uint64_t out_dim;
    const void *values;
    uint8_t *out;
    uint8_t *scratch;
    uint64_t scratch_bytes;
} PyCellPackDenseRow;

typedef int32_t (*PyCellPackCsrFn)(const PyCellPackCsrRow *);
typedef int32_t (*PyCellPackDenseFn)(const PyCellPackDenseRow *);
typedef struct {
    uint32_t abi_version, struct_size;
    PyCellPackCsrFn csr;
    PyCellPackDenseFn dense;
    const void *user_data;
} PyCellPackProcessVTable;
#endif
