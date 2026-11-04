// Geneformer rank value encoding.
//
// The cell's expressed genes that have a token are ranked by expression
// divided by the gene's nonzero median across the pretraining corpus
// (Geneformer's TranscriptomeTokenizer); its scaling by the cell's total
// counts is common to every gene of the cell and leaves the ranking as it
// is. Ties go to the lower output dim. The top genes fill the sentence,
// between `<cls>` and `<eos>` when `special_tokens` (Geneformer V2) or alone
// (V1); the rest is padding. No draw is made: a cell's sentence is a
// function of its row.
//
//   #include "template/geneformer.hpp"
//   struct V1 : pycellpack::preset::GeneformerDefaults {
//       static constexpr uint32_t length = 2048;
//       static constexpr bool special_tokens = false;
//   };
//   PYCELLPACK_EXPORT_PROCESS_FN(geneformer_v1, pycellpack::preset::Geneformer<V1>)
//
// user_data points at `GeneformerTables`: per output dim the token (< 0:
// none) and the gene's median (<= 0: none). The schema is float32 with
// row_bytes `Geneformer<Config>::out_bytes`: int64 ids[length] and uint32
// length, the number of tokens before the padding.
#ifndef PYCELLPACK_TEMPLATE_GENEFORMER_HPP
#define PYCELLPACK_TEMPLATE_GENEFORMER_HPP

#include "common.hpp"

#include <algorithm>

namespace pycellpack {
namespace preset {

/// Runtime tables, C layout.
struct GeneformerTables {
    const int64_t *tokens;  // [n_genes] token per output dim, < 0: none
    const float *medians;   // [n_genes] nonzero median per output dim
    uint64_t n_genes;
    int64_t pad_token, cls_token, eos_token;
};

/// Default hyperparameters (Geneformer V2); derive and shadow to change them.
struct GeneformerDefaults {
    using value_type = float;
    static constexpr uint32_t length = 4096;        // model input size
    static constexpr bool special_tokens = true;    // <cls> ... <eos>
};

template <class P>
struct Geneformer {
    static constexpr uint32_t L = P::length;
    static constexpr uint32_t specials = P::special_tokens ? 2 : 0;
    static_assert(L > specials, "the sentence needs room for a gene");

    using value_type = typename P::value_type;
    using Tables = GeneformerTables;
    struct Output {
        int64_t ids[L];
        uint32_t length;
    };
    static constexpr size_t out_bytes = sizeof(Output);

    static int32_t process(const PyCellPackContext &, uint64_t out_dim, const Tables &t, Workspace &w,
                           uint8_t *out) {
        if (out_dim != t.n_genes) return kWrongOutDim;
        auto &e = w.entries;
        e.erase(std::remove_if(e.begin(), e.end(),
                               [&](const Entry &x) { return t.tokens[x.gene] < 0 || !(t.medians[x.gene] > 0); }),
                e.end());
        for (auto &x : e) x.value /= t.medians[x.gene];
        const auto order = [](const Entry &a, const Entry &b) {
            return a.value != b.value ? a.value > b.value : a.gene < b.gene;
        };
        const size_t n = std::min(e.size(), size_t(L - specials));
        if (n < e.size()) std::nth_element(e.begin(), e.begin() + n, e.end(), order);
        std::sort(e.begin(), e.begin() + n, order);

        auto &o = *reinterpret_cast<Output *>(out);
        std::fill(o.ids, o.ids + L, t.pad_token);
        size_t at = 0;
        if (P::special_tokens) o.ids[at++] = t.cls_token;
        for (size_t i = 0; i < n; ++i) o.ids[at++] = t.tokens[e[i].gene];
        if (P::special_tokens) o.ids[at++] = t.eos_token;
        o.length = uint32_t(at);
        return kOk;
    }
};

}  // namespace preset
}  // namespace pycellpack

#endif
