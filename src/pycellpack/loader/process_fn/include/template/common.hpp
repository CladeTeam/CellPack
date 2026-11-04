// Shared machinery of the model presets (template/<model>.hpp).
//
// A preset is a class template over a parameter struct: every training
// configuration (sequence length, sample sizes, special tokens, ...) is a
// compile-time constant of its own instantiation, so each configuration gets
// its own statically compiled process functions. Large per-gene tables (token
// ids, chromosome positions, ...) are runtime data passed as the ProcessFn's
// user data, with the C layout each preset documents.
//
// A preset `Model` provides
//   using value_type = float or double;   // the schema's dtype
//   using Tables = ...;                   // what user_data points at
//   static constexpr size_t out_bytes;    // the schema's row_bytes
//   static int32_t process(const PyCellPackContext &, uint64_t out_dim,
//                          const Tables &, Workspace &, uint8_t *out);
// and `ProcessFn<Model>` adapts it to the CSR and dense entry points:
//
//   #include "template/scgpt.hpp"
//   struct Config : pycellpack::preset::ScGptDefaults { static constexpr uint32_t length = 1200; };
//   PYCELLPACK_EXPORT_PROCESS_FN(scgpt_1200, pycellpack::preset::ScGpt<Config>)
//
// exports `scgpt_1200_csr` and `scgpt_1200_dense`, to bind with
// `ProcessFn.from_source(code=..., csr="scgpt_1200_csr", dense="scgpt_1200_dense",
// user_data=tables)` and an OutputSchema of `row_bytes = out_bytes`.
//
// Rows reach `process` as the cell's positive entries sorted by output dim,
// whatever order the codec stored them in, so random draws do not depend on
// the storage. Every function is thread-safe: scratch is thread-local and no
// row pointer outlives the call.
#ifndef PYCELLPACK_TEMPLATE_COMMON_HPP
#define PYCELLPACK_TEMPLATE_COMMON_HPP

#include "../process_fn_cabi.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace pycellpack {
namespace preset {

/// Status codes shared by every preset; presets add their own below -100.
enum Status : int32_t {
    kOk = 0,
    kWrongDtype = -1,    // the schema's dtype is not the preset's value_type
    kPartialRow = -2,    // a CSR row arrived without columns/values
    kNoTables = -3,      // user_data is NULL
    kWrongOutDim = -4,   // the row's out_dim differs from the tables'
    kException = -99,    // a C++ exception (allocation) inside the call
};

template <class T> constexpr uint32_t dtype_code();
template <> constexpr uint32_t dtype_code<float>() { return 8; }
template <> constexpr uint32_t dtype_code<double>() { return 9; }

/// SplitMix64 finalizer: a well-mixed 64-bit hash.
inline uint64_t mix(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

/// A counter-based generator; `below` is unbiased by rejection.
struct Rng {
    uint64_t state;
    uint64_t next() { return mix(state += UINT64_C(0x9e3779b97f4a7c15)); }
    uint64_t below(uint64_t n) {
        uint64_t x = next();
        if (x < n) {
            const uint64_t threshold = -n % n;
            while (x < threshold) x = next();
        }
        return x % n;
    }
    double unit() { return (next() >> 11) * 0x1.0p-53; }
};

/// One expressed gene of the row: its output dim, value and a tie-break draw.
struct Entry {
    uint32_t gene;
    double value;
    uint64_t tie;
};

/// Thread-local scratch every preset shares; vectors keep their capacity
/// between calls.
struct Workspace {
    std::vector<Entry> entries, radix, valid, ranked;
    std::vector<double> sorted, cdf, values;
    std::vector<uint32_t> expressed, excluded, sampled, histogram, picked, value_stamp;
    std::vector<uint8_t> hidden;
    std::vector<std::pair<size_t, size_t>> groups;
    std::array<std::vector<uint32_t>, 8> choices;
    size_t choice = 0;
    uint32_t pick_epoch = 0, value_epoch = 0;
    // A per-microbatch draw shared by the cells of one batch (STATE SE).
    std::vector<int64_t> shared;
    uint64_t shared_key = 0;
    bool shared_valid = false;

    /// A fresh stamp for a stamped set over `n` ids, clearing on wrap-around.
    uint32_t epoch(std::vector<uint32_t> &stamps, uint32_t &counter, size_t n) {
        if (stamps.size() < n) stamps.resize(n, 0);
        if (++counter == 0) {
            std::fill(stamps.begin(), stamps.end(), 0);
            counter = 1;
        }
        return counter;
    }

    /// `k` distinct draws below `n` in random order (Floyd's algorithm, then
    /// a shuffle), or `k` draws with replacement. At most eight results are
    /// alive per call of a process function.
    const std::vector<uint32_t> &choose(uint32_t n, uint32_t k, Rng &r, bool replace) {
        auto &result = choices.at(choice++);
        result.clear();
        result.reserve(k);
        if (!n) return result;
        if (replace) {
            for (uint32_t i = 0; i < k; ++i) result.push_back(uint32_t(r.below(n)));
        } else {
            const uint32_t stamp = epoch(picked, pick_epoch, n);
            for (uint32_t j = n - k; j < n; ++j) {
                uint32_t v = uint32_t(r.below(uint64_t(j) + 1));
                if (picked[v] == stamp) v = j;
                picked[v] = stamp;
                result.push_back(v);
            }
            for (size_t i = result.size(); i > 1; --i) std::swap(result[i - 1], result[r.below(i)]);
        }
        return result;
    }
};

/// The `rank`-th id (ascending) missing from the sorted `excluded` ids.
inline uint32_t complement(uint32_t rank, const std::vector<uint32_t> &excluded) {
    size_t lo = 0, hi = excluded.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (excluded[mid] - mid <= rank) lo = mid + 1;
        else hi = mid;
    }
    return uint32_t(rank + lo);
}

/// Sorts entries by gene: a two-pass radix sort for large rows of a narrow
/// panel, a comparison sort otherwise; already sorted rows are kept.
inline void sort_by_gene(std::vector<Entry> &e, std::vector<Entry> &temp, uint64_t out_dim) {
    const auto by_gene = [](const Entry &a, const Entry &b) { return a.gene < b.gene; };
    if (std::is_sorted(e.begin(), e.end(), by_gene)) return;
    if (e.size() >= 256 && out_dim <= 65536) {
        temp.resize(e.size());
        for (unsigned shift : {0U, 8U}) {
            std::array<size_t, 256> counts{};
            for (auto &x : e) ++counts[(x.gene >> shift) & 255];
            size_t sum = 0;
            for (auto &count : counts) {
                const size_t n = count;
                count = sum;
                sum += n;
            }
            for (auto &x : e) temp[counts[(x.gene >> shift) & 255]++] = x;
            e.swap(temp);
        }
    } else {
        std::sort(e.begin(), e.end(), by_gene);
    }
}

/// Adapts a preset to the two entry points of a process vtable.
template <class Model>
struct ProcessFn {
    using value_type = typename Model::value_type;
    using Tables = typename Model::Tables;

    static int32_t csr(const PyCellPackCsrRow *row) noexcept {
        try {
            if (row->context.dtype != dtype_code<value_type>()) return kWrongDtype;
            if (row->nnz && (!row->values || !row->columns)) return kPartialRow;
            if (!row->context.user_data) return kNoTables;
            Workspace &w = begin();
            const auto *v = static_cast<const value_type *>(row->values);
            for (uint64_t i = 0; i < row->nnz; ++i)
                if (v[i] > 0) w.entries.push_back({row->columns[i], double(v[i]), 0});
            sort_by_gene(w.entries, w.radix, row->out_dim);
            return Model::process(row->context, row->out_dim, tables(row->context), w, row->out);
        } catch (...) {
            return kException;
        }
    }

    static int32_t dense(const PyCellPackDenseRow *row) noexcept {
        try {
            if (row->context.dtype != dtype_code<value_type>()) return kWrongDtype;
            if (!row->context.user_data) return kNoTables;
            Workspace &w = begin();
            const auto *v = static_cast<const value_type *>(row->values);
            for (uint64_t i = 0; i < row->out_dim; ++i)
                if (v[i] > 0) w.entries.push_back({uint32_t(i), double(v[i]), 0});
            return Model::process(row->context, row->out_dim, tables(row->context), w, row->out);
        } catch (...) {
            return kException;
        }
    }

private:
    static const Tables &tables(const PyCellPackContext &context) {
        return *static_cast<const Tables *>(context.user_data);
    }
    static Workspace &begin() {
        static thread_local Workspace w;
        w.choice = 0;
        w.entries.clear();
        return w;
    }
};

}  // namespace preset
}  // namespace pycellpack

/// Exports `<prefix>_csr` and `<prefix>_dense` for the preset instantiation
/// given as the remaining arguments (commas in template arguments are fine).
#define PYCELLPACK_EXPORT_PROCESS_FN(prefix, ...)                                         \
    extern "C" int32_t prefix##_csr(const PyCellPackCsrRow *row) {                        \
        return ::pycellpack::preset::ProcessFn<__VA_ARGS__>::csr(row);                    \
    }                                                                                     \
    extern "C" int32_t prefix##_dense(const PyCellPackDenseRow *row) {                    \
        return ::pycellpack::preset::ProcessFn<__VA_ARGS__>::dense(row);                  \
    }

#endif
