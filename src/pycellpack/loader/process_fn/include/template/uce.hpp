// UCE cell sentences: masked expression-weighted gene sampling grouped by
// chromosome, with positive and negative genes for the expression loss.
//
// A `mask_ratio` share of the cell's expressed genes is hidden; `sample`
// genes are drawn with replacement from the rest, weighted by log1p of their
// expression, sorted by chromosome and start, and written as chromosome
// groups in random order (`chrom_offset + chromosome`, the group's genes,
// `chrom_close_token`) after `<cls>`. Positives are hidden genes (any
// expressed gene once those run out); negatives are panel genes the cell does
// not express. Draws depend on the seed and the global cell id only.
//
//   #include "template/uce.hpp"
//   struct Config : pycellpack::preset::UceDefaults { static constexpr uint32_t sample = 1024; };
//   PYCELLPACK_EXPORT_PROCESS_FN(uce, pycellpack::preset::Uce<Config>)
//
// user_data points at `UceTables`. Genes without a protein token (tokens < 0)
// are ignored; `valid` lists the output dims that have one, in ascending
// order, and `local[g]` is g's index in `valid`. The schema is float32 with
// row_bytes `Uce<Config>::out_bytes`: int64 ids[length], int64
// loss_ids[positive + negative], float labels[positive + negative] (1 for
// positives, 0 for negatives) and uint32 length, the sentence's used length.
#ifndef PYCELLPACK_TEMPLATE_UCE_HPP
#define PYCELLPACK_TEMPLATE_UCE_HPP

#include "common.hpp"

#include <algorithm>
#include <cmath>

namespace pycellpack {
namespace preset {

/// Runtime tables, C layout.
struct UceTables {
    const int64_t *tokens;  // [n_genes] protein token per output dim, < 0: none
    const int64_t *chroms;  // [n_genes] chromosome id per output dim
    const int64_t *starts;  // [n_genes] genomic start per output dim
    const uint32_t *valid;  // [n_valid] output dims with a token, ascending
    const uint32_t *local;  // [n_genes] index into valid (for valid dims)
    uint64_t n_genes, n_valid;
    int64_t pad_token, cls_token, chrom_offset, chrom_close_token;
    uint64_t seed;
};

/// Default hyperparameters; derive and shadow to change them.
struct UceDefaults {
    using value_type = float;
    static constexpr uint32_t length = 1536;   // sentence capacity
    static constexpr uint32_t sample = 1024;   // sampled genes
    static constexpr uint32_t positive = 512;  // positive loss genes
    static constexpr uint32_t negative = 512;  // negative loss genes
    static constexpr double mask_ratio = 0.2;  // hidden share of expressed genes
};

enum UceStatus : int32_t {
    kUceTooFewGenes = -111,  // fewer than two expressed genes with tokens
    kUceAllMasked = -112,    // the mask hides every expressed gene
    kUceTooLong = -113,      // the sentence exceeds `length`
};

template <class P>
struct Uce {
    static constexpr uint32_t L = P::length, Q = P::positive + P::negative;
    static_assert(L >= 2, "a sentence needs <cls> and one gene");

    using value_type = typename P::value_type;
    using Tables = UceTables;
    struct Output {
        int64_t ids[L];
        int64_t loss_ids[Q];
        float labels[Q];
        uint32_t length;
    };
    static constexpr size_t out_bytes = sizeof(Output);

    static int32_t process(const PyCellPackContext &context, uint64_t out_dim, const Tables &t,
                           Workspace &w, uint8_t *out) {
        if (out_dim != t.n_genes) return kWrongOutDim;
        Rng r{mix(t.seed ^ mix(context.cell + UINT64_C(0x63656c6c)) ^ mix(1))};
        auto &e = w.entries;
        e.erase(std::remove_if(e.begin(), e.end(), [&](const Entry &x) { return t.tokens[x.gene] < 0; }), e.end());
        if (e.size() < 2) return kUceTooFewGenes;

        auto &o = *reinterpret_cast<Output *>(out);
        std::fill(o.ids, o.ids + L, t.pad_token);
        auto &expressed = w.expressed;
        expressed.clear();
        for (auto &x : e) expressed.push_back(t.local[x.gene]);
        const uint32_t nm = uint32_t(std::max(1L, std::lrint(P::mask_ratio * e.size())));
        if (nm >= e.size()) return kUceAllMasked;
        const auto &masked = w.choose(uint32_t(e.size()), nm, r, false);
        auto &hidden = w.hidden;
        hidden.assign(e.size(), 0);
        for (auto i : masked) hidden[i] = 1;

        // Expression-weighted sampling of the visible genes.
        auto &cdf = w.cdf;
        cdf.clear();
        double total = 0;
        for (size_t i = 0; i < e.size(); ++i) {
            total += hidden[i] ? 0 : std::log1p(e[i].value);
            cdf.push_back(total);
        }
        auto &sampled = w.sampled;
        sampled.clear();
        for (uint32_t i = 0; i < P::sample; ++i) {
            const size_t j = size_t(std::upper_bound(cdf.begin(), cdf.end(), r.unit() * total) - cdf.begin());
            sampled.push_back(e[std::min(j, e.size() - 1)].gene);
        }
        std::sort(sampled.begin(), sampled.end(), [&](uint32_t a, uint32_t b) {
            return t.chroms[a] != t.chroms[b] ? t.chroms[a] < t.chroms[b] : t.starts[a] < t.starts[b];
        });
        auto &groups = w.groups;
        groups.clear();
        for (size_t i = 0, j; i < sampled.size(); i = j) {
            j = i + 1;
            while (j < sampled.size() && t.chroms[sampled[j]] == t.chroms[sampled[i]]) ++j;
            groups.emplace_back(i, j);
        }
        const auto &order = w.choose(uint32_t(groups.size()), uint32_t(groups.size()), r, false);
        size_t at = 1;
        o.ids[0] = t.cls_token;
        for (auto k : order) {
            const auto [a, b] = groups[k];
            if (at + b - a + 2 > L) return kUceTooLong;
            o.ids[at++] = t.chrom_offset + t.chroms[sampled[a]];
            for (size_t j = a; j < b; ++j) o.ids[at++] = t.tokens[sampled[j]];
            o.ids[at++] = t.chrom_close_token;
        }
        o.length = uint32_t(at);

        // Positives: hidden genes first.
        const auto &pos = w.choose(uint32_t(masked.size()), std::min(P::positive, uint32_t(masked.size())), r, false);
        for (uint32_t i = 0; i < P::positive; ++i) {
            const auto j = i < pos.size() ? masked[pos[i]] : r.below(e.size());
            o.loss_ids[i] = t.tokens[e[j].gene];
            o.labels[i] = 1;
        }
        // Negatives: panel genes the cell does not express.
        const uint32_t unexpressed = uint32_t(t.n_valid - expressed.size());
        if (!unexpressed) {
            for (uint32_t i = 0; i < P::negative; ++i) {
                o.loss_ids[P::positive + i] = t.tokens[e[r.below(e.size())].gene];
                o.labels[P::positive + i] = 1;
            }
            return kOk;
        }
        const auto &neg = w.choose(unexpressed, P::negative, r, unexpressed < P::negative);
        for (uint32_t i = 0; i < P::negative; ++i) {
            o.loss_ids[P::positive + i] = t.tokens[t.valid[complement(neg[i], expressed)]];
            o.labels[P::positive + i] = 0;
        }
        return kOk;
    }
};

}  // namespace preset
}  // namespace pycellpack

#endif
