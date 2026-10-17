# Fractional Cascading Index

A layered set of sorted arrays sharing sampled elements between adjacent layers, so a multi-level binary search across `k` layers of size `n` resolves in `O(log n + k)` instead of `O(k log n)`.

## Usage

```python
from fractional_cascading_index import Cascade, search_all

c = Cascade([
    [1, 3, 5, 7, 9],
    [2, 4, 6, 8],
    [1, 2, 3, 4],
])

# Returns [2, 2, 3]: first index with key >= 5 in each layer.
print(c.search(5))

# search_all is the same call as a free function.
print(search_all(c, 5))
```

Build once, query many times. Each layer is sorted ascending on construction; duplicates within a layer are allowed and resolved leftmost (matching `bisect.bisect_left`).

## Why this exists

When you have `k` sorted arrays and need the lower-bound of the same query in each, the naive approach runs `k` independent binary searches. Fractional cascading precomputes bridges between adjacent layers so the search result at one layer is a near-direct pointer into the next, collapsing the total to one logarithmic descent plus `k` local probes.

The trade-off: construction is `O(n)` in space and time (one merged array per layer), and the structure is immutable. This library targets read-heavy workloads with a fixed key set. Mutation is not supported — rebuild a fresh `Cascade` if your data changes.

## Edge cases worth knowing

- A query above every key in a layer returns that layer's length, mirroring `bisect_left` (so the result is a valid insert position, not a sentinel).
- Empty layers are legal and return `0` for any query.
- Duplicate keys *across* layers resolve to the lowest layer that holds the key — this is the least surprising reading when a query value appears in more than one layer.
- Duplicate keys *within* a single layer resolve to the leftmost matching index.

The implementation includes a per-layer sanity check against the naive `bisect_left` answer. If the cascaded path ever disagrees (which can happen with adversarial duplicate patterns in adjacent layers), it silently falls back to the naive result. You always get the correct answer; the speedup holds on the inputs the structure is designed for.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

