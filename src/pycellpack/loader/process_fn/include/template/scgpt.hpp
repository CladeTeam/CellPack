// scGPT cell sentences: value binning, gene sampling and value masking.
//
// The cell's expressed genes that have a vocabulary token are binned by the
// cell's own quantiles (scGPT `binning`: `n_bins - 1` quantile edges over the
// nonzero values, ties broken by a uniform draw), sampled without replacement
// down to `length - 1` genes, and written after `<cls>`; masked positions
// carry `mask_value` as input and keep the bin as target. Draws depend on the
// seed and the global cell id only, so a cell keeps its sentence whatever the
// access order.
//
//   #include "template/scgpt.hpp"
//   struct Config : pycellpack::preset::ScGptDefaults {
//       static constexpr uint32_t length = 1200;
//       static constexpr double mask_ratio = 0.15;
//   };
//   PYCELLPACK_EXPORT_PROCESS_FN(scgpt, pycellpack::preset::ScGpt<Config>)
//
// user_data points at `ScGptTables` (as a ctypes.Structure): per output dim
// the vocabulary token (negative: gene not in the vocabulary), and the
// special tokens. The schema is float32 (`value_type`) with row_bytes
// `ScGpt<Config>::out_bytes`: int64 genes[length], float values[length],
// float targets[length], unused positions holding pad_token / pad_value.
#ifndef PYCELLPACK_TEMPLATE_SCGPT_HPP
#define PYCELLPACK_TEMPLATE_SCGPT_HPP

#include "common.hpp"

#include <algorithm>
#include <cmath>

namespace pycellpack {
namespace preset {

/// Runtime tables, C layout.
struct ScGptTables {
    const int64_t *tokens;  // [n_genes] vocabulary token per output dim, < 0: none
    uint64_t n_genes;       // the schema's out_dim
    int64_t pad_token, cls_token;
    uint64_t seed;
};

/// Default hyperparameters; derive and shadow to change them.
struct ScGptDefaults {
    using value_type = float;
    static constexpr uint32_t length = 1024;  // sequence length, <cls> included
    static constexpr uint32_t n_bins = 51;    // bins 1..n_bins-1 for nonzero values
    static constexpr double mask_ratio = 0.4; // share of genes whose input value is masked
    static constexpr float pad_value = -2.f;
    static constexpr float mask_value = -1.f;
    static constexpr float cls_value = -2.f;
};

enum ScGptStatus : int32_t { kScGptNoGenes = -101 };

template <class P>
struct ScGpt {
    static_assert(P::length >= 2, "a sentence needs <cls> and one gene");
    static_assert(P::n_bins >= 3, "binning needs at least two quantile edges");
    static constexpr uint32_t L = P::length;
    static constexpr uint32_t edges = P::n_bins - 1;

    using value_type = typename P::value_type;
    using Tables = ScGptTables;
    struct Output {
        int64_t genes[L];
        float values[L];
        float targets[L];
    };
    static constexpr size_t out_bytes = sizeof(Output);

    static int32_t process(const PyCellPackContext &context, uint64_t out_dim, const Tables &t,
                           Workspace &w, uint8_t *out) {
        if (out_dim != t.n_genes) return kWrongOutDim;
        Rng r{mix(t.seed ^ mix(context.cell + UINT64_C(0x63656c6c)) ^ mix(0))};
        auto &e = w.entries;
        e.erase(std::remove_if(e.begin(), e.end(), [&](const Entry &x) { return t.tokens[x.gene] < 0; }), e.end());
        if (e.empty()) return kScGptNoGenes;

        auto &o = *reinterpret_cast<Output *>(out);
        std::fill(o.genes, o.genes + L, t.pad_token);
        std::fill(o.values, o.values + L, P::pad_value);
        std::fill(o.targets, o.targets + L, P::pad_value);
        o.genes[0] = t.cls_token;
        o.values[0] = o.targets[0] = P::cls_value;

        // The cell's sorted values: a histogram for small integer counts.
        auto &sorted = w.sorted;
        sorted.resize(e.size());
        uint32_t maximum = 0;
        bool integer = true;
        for (auto &x : e) {
            if (x.value > 4096 || x.value != std::floor(x.value)) {
                integer = false;
                break;
            }
            maximum = std::max(maximum, uint32_t(x.value));
        }
        if (integer && maximum <= 2 * e.size()) {
            auto &histogram = w.histogram;
            histogram.assign(maximum + 1, 0);
            for (auto &x : e) ++histogram[uint32_t(x.value)];
            size_t at = 0;
            for (uint32_t v = 0; v <= maximum; ++v) {
                std::fill_n(sorted.begin() + at, histogram[v], double(v));
                at += histogram[v];
            }
        } else {
            for (size_t i = 0; i < e.size(); ++i) sorted[i] = e[i].value;
            std::sort(sorted.begin(), sorted.end());
        }
        double bins[edges];
        for (uint32_t i = 0; i < edges; ++i) {
            const double p = (sorted.size() - 1) * (double(i) / (edges - 1));
            const size_t j = size_t(p);
            bins[i] = sorted[j] + (sorted[std::min(j + 1, sorted.size() - 1)] - sorted[j]) * (p - j);
        }
        // Every gene draws its tie-break, selected or not, so the stream of
        // draws does not depend on which genes are kept.
        for (auto &x : e) x.tie = r.next();
        const size_t n = std::min(e.size(), size_t(L - 1));
        const std::vector<uint32_t> *selected = nullptr;
        if (e.size() > L - 1) selected = &w.choose(uint32_t(e.size()), uint32_t(n), r, false);
        for (size_t i = 0; i < n; ++i) {
            const Entry &x = e[selected ? (*selected)[i] : i];
            const int left = int(std::upper_bound(bins, bins + edges, x.value) - bins);
            const int right = int(std::lower_bound(bins, bins + edges, x.value) - bins);
            const double binned = std::ceil(((x.tie >> 11) * 0x1.0p-53) * (right - left) + left);
            o.genes[i + 1] = t.tokens[x.gene];
            o.targets[i + 1] = float(binned);
            o.values[i + 1] = r.unit() < P::mask_ratio ? P::mask_value : float(binned);
        }
        return kOk;
    }
};

}  // namespace preset
}  // namespace pycellpack

#endif
