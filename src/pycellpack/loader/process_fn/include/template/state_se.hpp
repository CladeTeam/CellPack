// STATE SE (state embedding) cell sentences and gene expression queries.
//
// Values are log1p'd when the row holds raw counts (`raw_counts`: 1 always,
// 2 never, 0 detected as upstream does: a maximum above 35 or an expm1 sum
// above 5e6). Position 0 holds the token of the panel gene at `cls_index`
// (its embedding is replaced by the learned CLS); the cell's genes follow in
// descending expression (ties broken at random) up to `length - 1`, the rest
// filled with unexpressed panel genes drawn at random; `counts` holds each
// position's share of the cell's total, in percent. The queries are
// `positive` expressed genes, `negative` genes below 1 and `shared` protein
// tokens drawn once per microbatch of `batch` consecutive positions (the
// collator's shared draw), each with its target value; `total` is the sum
// of the targets. Per-cell draws depend on the seed and the global cell id;
// the shared draw on the seed, `stream` and the microbatch.
//
//   #include "template/state_se.hpp"
//   struct Config : pycellpack::preset::StateSeDefaults { static constexpr uint32_t batch = 128; };
//   PYCELLPACK_EXPORT_PROCESS_FN(state_se, pycellpack::preset::StateSe<Config>)
//
// user_data points at `StateSeTables`: per output dim the protein token (< 0:
// none), `valid` (output dims with a token, ascending, the panel), `local`
// (index into valid) and `reverse` (protein token -> output dim, -1 for
// none; with duplicate tokens the last dim). The schema is float32 with
// row_bytes `StateSe<Config>::out_bytes`: int64 ids[length], int64
// tasks[Q], float targets[Q], float counts[length], float total, where
// Q = positive + negative + shared.
#ifndef PYCELLPACK_TEMPLATE_STATE_SE_HPP
#define PYCELLPACK_TEMPLATE_STATE_SE_HPP

#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace pycellpack {
namespace preset {

/// Runtime tables, C layout.
struct StateSeTables {
    const int64_t *tokens;   // [n_genes] protein token per output dim, < 0: none
    const uint32_t *valid;   // [n_valid] output dims with a token, ascending
    const uint32_t *local;   // [n_genes] index into valid (for valid dims)
    const int64_t *reverse;  // [global_size] protein token -> output dim, -1: none
    uint64_t n_genes, n_valid, global_size;
    uint64_t seed;
    uint32_t stream;         // separates shared draws of concurrent streams
};

/// Default hyperparameters; derive and shadow to change them.
struct StateSeDefaults {
    using value_type = float;
    static constexpr uint32_t length = 2048;    // sentence length, CLS included
    static constexpr uint32_t positive = 512;   // expressed query genes
    static constexpr uint32_t negative = 512;   // query genes below 1
    static constexpr uint32_t shared = 512;     // shared query tokens per microbatch
    static constexpr uint32_t batch = 128;      // positions per microbatch
    static constexpr uint32_t raw_counts = 0;   // 0 detect, 1 raw counts, 2 log values
    static constexpr uint32_t cls_index = 3;    // panel gene whose token fills position 0
};

enum StateSeStatus : int32_t {
    kStateSeSmallPanel = -121,  // the panel cannot hold cls_index
    kStateSeFullPanel = -122,   // no unexpressed gene left to pad the sentence
    kStateSeQueries = -123,     // too few genes for the queries
};

template <class P>
struct StateSe {
    static constexpr uint32_t L = P::length, Q = P::positive + P::negative + P::shared;
    static_assert(L >= 2, "a sentence needs CLS and one gene");
    static_assert(P::batch >= 1, "a microbatch holds at least one position");

    using value_type = typename P::value_type;
    using Tables = StateSeTables;
    struct Output {
        int64_t ids[L];
        int64_t tasks[Q];
        float targets[Q];
        float counts[L];
        float total;
    };
    static constexpr size_t out_bytes = sizeof(Output);

    static int32_t process(const PyCellPackContext &context, uint64_t out_dim, const Tables &t,
                           Workspace &w, uint8_t *out) {
        if (out_dim != t.n_genes) return kWrongOutDim;
        const uint32_t *panel = t.valid;
        const size_t panel_size = t.n_valid;
        if (panel_size <= P::cls_index) return kStateSeSmallPanel;
        Rng r{mix(t.seed ^ mix(context.cell + UINT64_C(0x63656c6c)) ^ mix(2))};
        auto &e = w.entries;
        auto &o = *reinterpret_cast<Output *>(out);

        bool raw = P::raw_counts == 1;
        if (P::raw_counts == 0) {
            double maximum = 0, inverse_sum = 0;
            for (auto &x : e) maximum = std::max(maximum, x.value);
            if (maximum > 35) {
                raw = true;
            } else {
                for (auto &x : e) inverse_sum += std::expm1(x.value);
                raw = inverse_sum > 5000000;
            }
        }
        auto &valid = w.valid;
        valid.clear();
        for (auto x : e) {
            if (t.tokens[x.gene] >= 0) {
                x.value = raw ? std::log1p(x.value) : x.value;
                x.tie = r.next();
                valid.push_back(x);
            }
        }
        const uint32_t value_epoch = w.epoch(w.value_stamp, w.value_epoch, t.n_genes);
        w.values.resize(t.n_genes);
        for (auto &x : valid) {
            w.values[x.gene] = x.value;
            w.value_stamp[x.gene] = value_epoch;
        }
        const auto value = [&](uint32_t gene) {
            return w.value_stamp[gene] == value_epoch ? w.values[gene] : 0.;
        };
        auto &expressed = w.expressed;
        auto &excluded = w.excluded;
        expressed.clear();
        excluded.clear();
        double sum = 0;
        for (auto &x : valid) {
            const uint32_t local = t.local[x.gene];
            expressed.push_back(local);
            if (x.value >= 1) excluded.push_back(local);
            sum += x.value;
        }

        // The sentence: genes by descending expression, then unexpressed ones.
        auto &ranked = w.ranked;
        ranked.assign(valid.begin(), valid.end());
        const size_t n = std::min(ranked.size(), size_t(L - 1));
        const auto rank_order = [](const Entry &a, const Entry &b) {
            return a.value != b.value ? a.value > b.value : a.tie < b.tie;
        };
        if (n < ranked.size()) std::nth_element(ranked.begin(), ranked.begin() + n, ranked.end(), rank_order);
        std::sort(ranked.begin(), ranked.begin() + n, rank_order);
        o.ids[0] = t.tokens[panel[P::cls_index]];
        o.counts[0] = float(sum > 0 ? 100 * value(panel[P::cls_index]) / sum : 100. / panel_size);
        for (size_t i = 0; i < n; ++i) {
            o.ids[i + 1] = t.tokens[ranked[i].gene];
            o.counts[i + 1] = float(100 * ranked[i].value / sum);
        }
        const uint32_t remaining = uint32_t(L - 1 - n);
        if (remaining) {
            const uint32_t zeros = uint32_t(panel_size - expressed.size());
            if (!zeros) return kStateSeFullPanel;
            const auto &z = w.choose(zeros, remaining, r, panel_size < L - 1);
            for (uint32_t i = 0; i < remaining; ++i) {
                o.ids[1 + n + i] = t.tokens[panel[complement(z[i], expressed)]];
                o.counts[1 + n + i] = float(sum > 0 ? 0 : 100. / panel_size);
            }
        }

        // Queries: positives, negatives, then the microbatch's shared tokens.
        const auto &p = w.choose(uint32_t(valid.size()), P::positive, r, valid.size() <= P::positive);
        const uint32_t below_one = uint32_t(panel_size - excluded.size());
        const auto &z = w.choose(below_one, P::negative, r, below_one <= P::negative);
        if ((!valid.empty() && p.size() != P::positive) || z.size() != P::negative) return kStateSeQueries;
        for (uint32_t i = 0; i < P::positive; ++i) {
            o.tasks[i] = valid.empty() ? t.tokens[panel[0]] : t.tokens[valid[p[i]].gene];
            o.targets[i] = float(valid.empty() ? 0 : valid[p[i]].value);
        }
        for (uint32_t i = 0; i < P::negative; ++i) {
            const uint32_t g = panel[complement(z[i], excluded)];
            o.tasks[P::positive + i] = t.tokens[g];
            o.targets[P::positive + i] = float(value(g));
        }
        // One shared draw per microbatch, independent of which worker runs it.
        const uint64_t key = mix(t.seed ^ mix(t.stream) ^ mix(context.position / P::batch) ^ UINT64_C(0x736861726564));
        if (!w.shared_valid || w.shared_key != key || w.shared.size() != P::shared) {
            w.shared.resize(P::shared);
            Rng draw{key};
            for (auto &token : w.shared) token = int64_t(draw.below(t.global_size));
            w.shared_key = key;
            w.shared_valid = true;
        }
        for (uint32_t i = 0; i < P::shared; ++i) {
            const int64_t token = w.shared[i];
            o.tasks[P::positive + P::negative + i] = token;
            o.targets[P::positive + P::negative + i] = float(t.reverse[token] >= 0 ? value(uint32_t(t.reverse[token])) : 0.);
        }
        o.total = float(std::accumulate(o.targets, o.targets + Q, 0.0));
        return kOk;
    }
};

}  // namespace preset
}  // namespace pycellpack

#endif
