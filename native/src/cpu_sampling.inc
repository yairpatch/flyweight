// Host drivers for the block top-k sampling kernels.
//
// Included by cpu_kernels.cpp after the generated corpus so the sort key is
// the corpus's own sampling_sort_key. Each corpus block sorts 1,024 candidates
// with cub::BlockRadixSort (emulated as a stable sort on one fiber after three
// block barriers) and keeps its top_k: ~15 ms of every sampled token on a
// 248k vocabulary. Here each block is a partial sort, run across the pool.
//
// The emulated sort is stable, and items enter it in candidate order, so ties
// -- which only the zero key of a NaN logit or a padding slot can produce --
// resolve to the earlier candidate. Ordering by (key descending, position
// ascending) reproduces that exactly for the ranks that are kept.

namespace sampling_topk {

constexpr int kBlockItems = 1024;

struct Candidate {
    unsigned long long key;
    int token;
    int position;
};

template <class Load>
void run(const ::flyweight::cpu::Launch& launch, int count, int top_k,
         int* output_indices, float* output_values, const Load& load) {
    const int keep = std::min(top_k, kBlockItems);
    if (keep <= 0) return;
    q8_family::parallel_ranges(launch.grid_x, [&](std::uint64_t first,
                                                  std::uint64_t last) {
        std::vector<Candidate> items(kBlockItems);
        for (std::uint64_t block = first; block < last; ++block) {
            for (int position = 0; position < kBlockItems; ++position) {
                const int input = static_cast<int>(block) * kBlockItems + position;
                Candidate& item = items[position];
                item.position = position;
                if (input < count) {
                    load(input, &item.token, &item.key);
                } else {
                    item.token = -1;
                    item.key = 0;
                }
            }
            std::partial_sort(items.begin(), items.begin() + keep, items.end(),
                              [](const Candidate& a, const Candidate& b) {
                                  if (a.key != b.key) return a.key > b.key;
                                  return a.position < b.position;
                              });
            for (int rank = 0; rank < keep; ++rank) {
                const long long output = static_cast<long long>(block) * top_k + rank;
                output_indices[output] = items[rank].token;
                output_values[output] = sampling_key_value(items[rank].key);
            }
        }
    });
}

//   sampling_block_topk_logits(const float* logits, int* output_indices,
//                              float* output_values, int count, int top_k)
void logits(const ::flyweight::cpu::Launch& launch, void** arguments) {
    using q8_family::argument;
    const auto* values = argument<const float*>(arguments, 0);
    run(launch, argument<int>(arguments, 3), argument<int>(arguments, 4),
        argument<int*>(arguments, 1), argument<float*>(arguments, 2),
        [&](int input, int* token, unsigned long long* key) {
            *token = input;
            *key = sampling_sort_key(values[input], input);
        });
}

//   sampling_block_topk_pairs(const int* input_indices,
//                             const float* input_values, int* output_indices,
//                             float* output_values, int count, int top_k)
void pairs(const ::flyweight::cpu::Launch& launch, void** arguments) {
    using q8_family::argument;
    const auto* indices = argument<const int*>(arguments, 0);
    const auto* values = argument<const float*>(arguments, 1);
    run(launch, argument<int>(arguments, 4), argument<int>(arguments, 5),
        argument<int*>(arguments, 2), argument<float*>(arguments, 3),
        [&](int input, int* token, unsigned long long* key) {
            *token = indices[input];
            *key = sampling_sort_key(values[input], *token);
        });
}

}  // namespace sampling_topk

FLYWEIGHT_CPU_Q8_REGISTER(sampling_block_topk_logits, sampling_topk::logits);
FLYWEIGHT_CPU_Q8_REGISTER(sampling_block_topk_pairs, sampling_topk::pairs);
