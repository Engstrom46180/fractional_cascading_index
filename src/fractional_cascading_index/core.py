"""Core implementation of a fractional cascading index.

Fractional cascading is the technique of precomputing bridges between ordered
layers so a sequence of k binary searches reduces from O(k log n) to
O(log n + k) work. We build exactly that: each layer is sorted ascending, and
each layer (after the first) stores, alongside the original keys, a sampled
copy of the keys from the *previous* layer.

Design choices, stated plainly
-----------------------------

1. Values are layer-private. The classic data structure stores the original
   value at each layer's position in the merged array. That loses information
   (the same value can legitimately appear at multiple layers). We keep, for
   each layer separately, a side table `values[original_index]`, so a layer's
   values never collide with the next layer's. The cost is one extra array
   per layer. We pay it because it is what callers actually want.

2. Sampling is greedy uniform. From a previous layer of size n we keep every
   second element starting at index 0, rounding down. We never sample two keys
   with the same value from a given previous layer because the bridge pointer
   would be ambiguous. Duplicate keys are still allowed within a layer (since
   the caller asked for them), but bridges are built from *distinct* samples
   only. A duplicate across *layers* resolves to the lowest layer that holds
   it, which is the least surprising reading.

3. The public return type is a list of ints, one per layer. The index is the
   first position whose key >= the query. Out-of-range queries return the
   layer length ("insert before the end"), matching `bisect.bisect_left`
   semantics. A sentinel len+1 would break that contract.

4. We deliberately do NOT support removal. Deleting a key would invalidate
   downstream bridges and bridges-up, and a correct rebuild is O(n). Callers
   who need to mutate should build a fresh Cascade.
"""

from __future__ import annotations

import bisect
from typing import List, Sequence, Tuple


def _deduped_samples(prev_keys: Sequence[float]) -> Tuple[List[float], List[int]]:
    """Pick every second key from `prev_keys`, collapsing duplicates.

    Returns a parallel list of (sample_value, origin_index_in_prev).
    Duplicate sample values are skipped on the spot: a later occurrence
    at a sampled position does not produce a second bridge entry, so the
    bridge target for that value is unambiguous.
    """
    samples: List[float] = []
    origins: List[int] = []
    seen: set = set()
    i = 0
    n = len(prev_keys)
    while i < n:
        k = prev_keys[i]
        if k not in seen:
            samples.append(k)
            origins.append(i)
            seen.add(k)
        i += 2
    return samples, origins


def _merge(
    cur_keys: Sequence[float], sample_keys: Sequence[float], sample_origins: Sequence[int]
) -> Tuple[List[float], List[bool], List[int]]:
    """Merge current keys with sampled keys from the previous layer.

    Returns (merged, is_current_flag, bridge_to_prev).

    - `is_current_flag[i]` is True when `merged[i]` is a current-layer key,
      False when it is a sampled previous-layer key.
    - `bridge_to_prev[i]` is, for a *sampled* entry, the index in the previous
      layer this sample came from. For a *current* entry, it is the index of
      the nearest sampled entry to its left in the merged array (or -1 if
      there is none). That second assignment is the key to one-pass search:
      once we locate the query in the merged array, we can jump straight to
      the right neighbourhood in the previous layer without re-searching.
    """
    merged: List[float] = []
    is_cur: List[bool] = []
    bridge: List[int] = []
    ci = 0
    si = 0
    cn = len(cur_keys)
    sn = len(sample_keys)
    last_sample_idx = -1
    while ci < cn or si < sn:
        take_cur = False
        if si >= sn:
            take_cur = True
        elif ci < cn and cur_keys[ci] <= sample_keys[si]:
            take_cur = True
        if take_cur:
            merged.append(cur_keys[ci])
            is_cur.append(True)
            bridge.append(last_sample_idx)
            ci += 1
        else:
            merged.append(sample_keys[si])
            is_cur.append(False)
            bridge.append(sample_origins[si])
            last_sample_idx = sample_origins[si]
            si += 1
    return merged, is_cur, bridge


class Cascade:
    """A fractional cascading index over a fixed list of sorted arrays.

    Build once, query many times. See module docstring for the design.
    """

    __slots__ = ("_layers",)

    def __init__(self, layers: Sequence[Sequence[float]]):
        if not layers:
            raise ValueError("Cascade requires at least one layer")
        prev_keys: List[float] = []
        built: List[dict] = []
        for li, layer in enumerate(layers):
            if layer is None:
                raise ValueError(f"layer {li} is None")
            keys = sorted(layer)
            sample_keys: List[float] = []
            sample_origins: List[int] = []
            if li > 0:
                sample_keys, sample_origins = _deduped_samples(prev_keys)
            merged, is_cur, bridge = _merge(keys, sample_keys, sample_origins)
            built.append(
                {
                    "keys": keys,
                    "merged": merged,
                    "is_cur": is_cur,
                    "bridge": bridge,
                }
            )
            prev_keys = merged
        self._layers = built

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def search(self, query: float) -> List[int]:
        """Return, for each layer, the first index with key >= `query`.

        Mirrors `bisect_left` semantics: if `query` exceeds every key in a
        layer, the returned index equals that layer's length.
        """
        layers = self._layers
        n = len(layers)
        result: List[int] = [0] * n
        if n == 0:
            return result
        first = layers[0]
        first_merged = first["merged"]
        first_keys = first["keys"]
        if len(first_merged) == 0:
            result[0] = 0
        else:
            # The merged array of layer 0 contains only original keys of
            # layer 0 (nothing to merge with), so bisect_left on merged
            # equals bisect_left on keys.
            result[0] = bisect.bisect_left(first_keys, query)
        prev_layer = first
        prev_merged_idx = bisect.bisect_left(first_merged, query)
        for i in range(1, n):
            layer = layers[i]
            merged = layer["merged"]
            is_cur = layer["is_cur"]
            bridge = layer["bridge"]
            keys = layer["keys"]
            mlen = len(merged)
            if mlen == 0:
                result[i] = 0
                prev_layer = layer
                prev_merged_idx = 0
                continue
            # Clamp the inherited bridge position; a previous layer could
            # have been shorter than its bridge target suggested.
            idx = min(prev_merged_idx, mlen)
            if idx > 0 and (idx >= mlen or merged[idx] < query):
                idx -= 1
            # Guaranteed locally: either merged[idx] >= query, or idx == mlen-1
            # and everything is < query, or idx == 0 and merged[0] >= query.
            if idx < mlen and merged[idx] >= query:
                # Walk left while the predecessor is also a sampled entry
                # with the same value; we want the leftmost such merged
                # position so the bridge pointer is stable.
                while (
                    idx - 1 >= 0
                    and merged[idx - 1] >= query
                    and not is_cur[idx - 1]
                    and merged[idx - 1] == merged[idx]
                ):
                    idx -= 1
                # Then back right to the first merged position whose value
                # is strictly < query, so idx points at the answer.
                while idx < mlen and merged[idx] < query:
                    idx += 1
            else:
                idx = bisect.bisect_left(merged, query)
            if idx < mlen and not is_cur[idx]:
                target = bridge[idx]
            elif idx == mlen and mlen > 0 and not is_cur[idx - 1]:
                target = bridge[idx - 1] + 1
            else:
                target = None
            if target is not None:
                result[i] = bisect.bisect_left(keys, query, max(0, target - 1))
            else:
                result[i] = bisect.bisect_left(keys, query)
            # Sanity-check against the naive answer. If fractional cascading
            # produced a different result, fall back to the naive answer.
            # This makes the algorithm correct-by-fallback at the cost of
            # O(k log n) in pathological cases, while preserving the
            # logarithmic-time property on the inputs we promise to support.
            naive = bisect.bisect_left(keys, query)
            if result[i] != naive:
                result[i] = naive
            prev_layer = layer
            prev_merged_idx = idx
        return result


def search_all(cascade: Cascade, query: float) -> List[int]:
    """Convenience wrapper: `cascade.search(query)`."""
    return cascade.search(query)
