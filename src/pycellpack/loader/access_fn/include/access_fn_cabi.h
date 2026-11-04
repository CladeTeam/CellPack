#ifndef CELLPACK_ACCESS_FN_CABI_H
#define CELLPACK_ACCESS_FN_CABI_H
#include <stdint.h>

#define CELLPACK_ACCESS_ABI_VERSION 1

/* Functions fill [start, start + count), return zero on success, and retain
 * no output pointers. Calls may be concurrent, repeated, and out of order.
 * Results depend only on position and immutable configuration. Exceptions
 * must not cross this C boundary. The provider keeps code and context alive.
 * The sequence length is supplied separately when binding the function. */
typedef int32_t (*CellPackAccessFn)(void *context, uint64_t start,
                                  uint64_t count, uint32_t *cells);

/* Dataset ids index the caller's corpus; rows index each dataset's current
 * row view. Both arrays have count entries and may be filled independently.
 * The adapter binds row counts and translates these pairs to global cells. */
typedef int32_t (*CellPackCorpusAccessFn)(void *context, uint64_t start,
                                        uint64_t count, uint32_t *dataset_ids,
                                        uint64_t *rows);

#if defined(_WIN32)
#define CELLPACK_ACCESS_EXPORT __declspec(dllexport)
#else
#define CELLPACK_ACCESS_EXPORT __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
#define CELLPACK_ACCESS_EXTERN extern "C"
#else
#define CELLPACK_ACCESS_EXTERN extern
#endif

#endif
