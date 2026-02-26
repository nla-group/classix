# CLASSIX_T — Tanimoto-Based CLASSIX Clustering

`classix_t.py` implements a variant of the CLASSIX clustering algorithm that uses the **Tanimoto distance** instead of the Euclidean distance used in the standard CLASSIX implementation (`classix/clustering.py`). This document describes how the merging phase and the `explain` feature in `CLASSIX_T` differ from the normal CLASSIX clustering.

---

## Overview of the Two-Phase CLASSIX Pipeline

Both the standard CLASSIX and CLASSIX_T follow a two-phase approach:

1. **Aggregation** — Data points are grouped around representative starting points.
2. **Merging** — Groups whose centers are close enough are merged into clusters.

The core algorithmic difference lies in **how distances are computed** during both phases and **how the merging graph is constructed and explained**.

---

## Merging Differences

### Distance Metric

| Aspect | Standard CLASSIX | CLASSIX_T |
|---|---|---|
| **Metric** | Euclidean (L2) distance | Tanimoto distance (`1 - inner_product / (‖a‖₁ + ‖b‖₁ - inner_product)`) |
| **Data representation** | Dense NumPy arrays | Sparse CSR matrices (`scipy.sparse.csr_matrix`) |
| **Sorting values** | Multiple options (PCA, norm-mean, norm-orthant) | Sum of features (`np.sum(data, axis=1)`) |

### Search Radius Calculation

The search window that limits which pairs of group centers are compared differs between the two implementations:

- **Standard CLASSIX (Euclidean):** `search_radius = mergeScale * radius + sort_val[i]`
- **CLASSIX_T (Tanimoto):** `search_radius = sort_val[i] / (1 - mergeScale * radius)`

In CLASSIX_T the Tanimoto-derived formula accounts for the fact that Tanimoto distance is bounded in `[0, 1]`, so the window is expressed as a multiplicative threshold rather than an additive one.

### Merging Strategies

Standard CLASSIX supports multiple merging methods:

- `"distance"` — merge groups whose centers are within `mergeScale * radius`
- `"density"` — merge based on connected components and density thresholds
- `"mst-distance"` and `"scc-distance"` — graph-based variants

**CLASSIX_T supports only one merging strategy: Tanimoto distance-based merging.** Groups are merged when:

```
tanimoto_distance(center_i, center_j) <= mergeScale * radius
```

### Adjacency Matrix

Both implementations build an adjacency matrix (`Adj`) to record which groups were merged:

- **Value `1`** — Groups connected by the primary distance-based merge.
- **Value `2`** — Groups connected during the **minPts redistribution** phase (small-cluster post-allocation).

In standard CLASSIX, the adjacency matrix is only constructed when using Tanimoto or Manhattan metrics. In CLASSIX_T, it is **always** constructed since the Tanimoto metric is the only option.

### Small-Cluster Redistribution (minPts)

After the initial merging, both implementations identify clusters with fewer than `minPts` total points and reallocate their constituent groups to the nearest valid cluster. The process is the same in principle:

1. Compute the Tanimoto (or Euclidean) distance from each orphaned group center to all other group centers.
2. Sort by distance and assign the group to the nearest cluster that meets the `minPts` threshold.
3. Mark the new edge in the adjacency matrix with value `2`.

### Default Parameters

| Parameter | Standard CLASSIX | CLASSIX_T |
|---|---|---|
| `mergeScale` | 1.5 | 1.4 |
| `radius` | 0.5 | 0.3 |

---

## The `explain` Feature

The `explain` method allows users to understand **why** specific data points were assigned to their clusters. CLASSIX_T provides a simplified, text-only version of this feature, while the standard CLASSIX offers a rich, visualization-heavy explanation.

### How `explain` Works in CLASSIX_T

The method signature is:

```python
explain(ind1=None, ind2=None)
```

There are three usage modes:

1. **No arguments** — Prints the number of groups and the number of final clusters:
   ```
   The data was clustered into 50 groups. These were further merged into 3 clusters.
   ```

2. **One index** — Prints which cluster the data point was assigned to:
   ```
   The data point at index 42 was assigned to cluster 1
   ```

3. **Two indices** — Explains whether two data points are in the same or different clusters. If they are in the same cluster but different groups, it uses **BFS (breadth-first search)** on the adjacency matrix to find the shortest path of group connections between them:
   ```
   The data points at indices 10 and 200 belong to the same cluster.
   The data points at indices 10 and 200 are in different groups.
   The connections between 10 and 200 are via this path: 3  -> 5 (minPts) -> 8
   ```

   The path annotation distinguishes between:
   - `->` — a normal distance-based merge (adjacency value `1`)
   - `(minPts) ->` — a connection created during the small-cluster redistribution phase (adjacency value `2`)

### Differences from Standard CLASSIX `explain`

| Feature | Standard CLASSIX | CLASSIX_T |
|---|---|---|
| **Visualization** | Rich 2D PCA-projected scatter plots with group circles, arrows, and customizable styling (40+ parameters) | None — console text output only |
| **Distance info** | Can include computed distances along the connection path | Not included |
| **Pandas support** | Supports DataFrame indices and `replace_name` for readable labels | Not supported |
| **Group details** | Shows group center coordinates, point counts, and cluster membership table | Not shown |
| **Path algorithm** | Uses shortest-distance path on a pairwise distance matrix | Uses BFS on the binary adjacency matrix |
| **Path annotation** | Does not annotate edge types; prints a separate message when no direct path exists due to minPts reassignment | Annotates each edge as `->` (distance merge) or `(minPts) ->` (redistribution), providing per-edge detail |
| **Customization** | Extensive: colors, markers, sizes, fonts, backgrounds, arrow styles, etc. | None |

### Why This Matters

The adjacency matrix with its dual encoding (`1` for distance merges, `2` for minPts-driven reallocations) is the key data structure that powers the `explain` feature in CLASSIX_T. By running BFS on this matrix and checking edge values, the `explain` method can trace and annotate the exact chain of group connections that caused two data points to end up in the same cluster — and it can tell the user **which connections were due to proximity and which were due to the minPts redistribution rule**.

This is a lightweight but effective explainability mechanism compared to the full visualization suite in standard CLASSIX.

---

## Summary

CLASSIX_T is a streamlined variant of CLASSIX designed for **sparse, binary, or count-based data** where the Tanimoto distance is the natural similarity measure. Its merging phase replaces Euclidean distance comparisons with Tanimoto distance computations over sparse CSR matrices, using an optimized sparse matrix-vector product (`spsubmatxvec`). The `explain` feature provides a text-based path explanation using BFS on an adjacency matrix that encodes both distance-based and minPts-based group connections, trading the rich visualizations of standard CLASSIX for simplicity and speed.
