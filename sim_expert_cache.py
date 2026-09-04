"""Replay a decode route trace against candidate expert-cache policies.

Reads the packed (uint32 layer, uint32 expert) stream written by
FLYWEIGHT_EXPERT_TRACE and reports the hit rate each policy would have achieved,
so a policy change can be justified before it is written in C++.

The device cache is statically partitioned per layer, so every policy is
simulated independently per layer with that layer's slot budget.
"""

import sys
from collections import Counter, defaultdict, OrderedDict

import numpy as np

TRACE = sys.argv[1] if len(sys.argv) > 1 else "routes.bin"
SLOTS = int(sys.argv[2]) if len(sys.argv) > 2 else 58
DECAY = 32768 // 40  # global decay period, apportioned to one layer


def lru(trace, slots):
    cache = OrderedDict()
    hits = 0
    for e in trace:
        if e in cache:
            cache.move_to_end(e)
            hits += 1
        else:
            if len(cache) >= slots:
                cache.popitem(last=False)
            cache[e] = True
    return hits


def lfu(trace, slots, strict, decay):
    """Frequency-ranked eviction; `strict` refuses to admit a candidate that is
    not hotter than the coldest resident, which is what the runtime does today."""
    freq = Counter()
    resident = {}
    hits = 0
    for i, e in enumerate(trace):
        freq[e] += 1
        if decay and i and i % decay == 0:
            for k in freq:
                if freq[k] > 1:
                    freq[k] = (freq[k] + 1) // 2
        if e in resident:
            resident[e] = i
            hits += 1
            continue
        if len(resident) < slots:
            resident[e] = i
            continue
        victim = min(resident, key=lambda k: (freq[k], resident[k]))
        if strict:
            if freq[e] <= freq[victim]:
                continue
        elif freq[e] < freq[victim]:
            continue
        del resident[victim]
        resident[e] = i
    return hits


def belady(trace, slots):
    """Oracle: evict whichever resident is next used farthest in the future."""
    nxt = defaultdict(list)
    for i, e in enumerate(trace):
        nxt[e].append(i)
    cursor = {e: 0 for e in nxt}
    resident = set()
    hits = 0
    for i, e in enumerate(trace):
        cursor[e] += 1
        if e in resident:
            hits += 1
            continue
        if len(resident) < slots:
            resident.add(e)
            continue

        def next_use(k):
            uses = nxt[k]
            c = cursor[k]
            return uses[c] if c < len(uses) else float("inf")

        victim = max(resident, key=next_use)
        if next_use(victim) == float("inf") or next_use(victim) > i:
            resident.discard(victim)
            resident.add(e)
    return hits


def main():
    raw = np.fromfile(TRACE, dtype=np.uint32).reshape(-1, 2)
    layers = raw[:, 0]
    experts = raw[:, 1]
    n_layers = int(layers.max()) + 1
    print(f"{len(raw)} routes, {n_layers} layers, {SLOTS} slots/layer, "
          f"{int(experts.max())+1} experts")

    per_layer = [experts[layers == l] for l in range(n_layers)]

    uniq = [len(set(t.tolist())) for t in per_layer]
    print(f"unique experts touched per layer: min {min(uniq)} "
          f"median {int(np.median(uniq))} max {max(uniq)}")

    totals = {}
    for name, fn in (
        ("current (LFU strict+decay)", lambda t: lfu(t, SLOTS, True, DECAY)),
        ("LFU relaxed+decay", lambda t: lfu(t, SLOTS, False, DECAY)),
        ("LFU strict, no decay", lambda t: lfu(t, SLOTS, True, 0)),
        ("LRU", lambda t: lru(t, SLOTS)),
        ("Belady (oracle)", lambda t: belady(t, SLOTS)),
    ):
        hits = sum(fn(t.tolist()) for t in per_layer)
        totals[name] = hits / len(raw)
        print(f"  {name:<28} hit {hits/len(raw):.4f}")

    # Slots are split evenly today; concentration tells us whether that is right.
    print("\nper-layer hit rate under current policy (even split):")
    rates = [lfu(t.tolist(), SLOTS, True, DECAY) / len(t) for t in per_layer]
    order = np.argsort(rates)
    worst = ", ".join(f"L{int(i)}={rates[i]:.2f}" for i in order[:5])
    best = ", ".join(f"L{int(i)}={rates[i]:.2f}" for i in order[-5:])
    print(f"  worst: {worst}")
    print(f"  best:  {best}")

    print("\nhit rate vs slots/layer (current policy):")
    for s in (32, 58, 96, 128, 192, 256):
        hits = sum(lfu(t.tolist(), s, True, DECAY) for t in per_layer)
        print(f"  {s:4d} slots -> {hits/len(raw):.4f}")


if __name__ == "__main__":
    main()
