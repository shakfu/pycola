# PyCola Performance Optimization Analysis

Based on profiling results from various graph sizes and configurations.

## Executive Summary

**✅ OPTIMIZATION COMPLETED**: Vectorized `compute_derivatives` achieving **20-170x speedup**

**Original Bottleneck**: The `compute_derivatives` method in `descent.py` accounted for **92-98%** of total runtime.

**Current Status** (after vectorization):
- ✅ **Phase 1 Complete**: NumPy vectorization implemented
- 🎯 **New Bottleneck**: Shortest path calculation (Dijkstra) now accounts for **75-82%** of runtime
- 📈 **Overall improvement**: 20-65x faster depending on graph size

**Original Findings**:
1. Gradient descent stress minimization dominated execution time (FIXED ✅)
2. O(n²) nested loops in derivative computation (FIXED ✅)
3. Frequent calls to `math.sqrt` and `math.isfinite` in hot paths (FIXED ✅)
4. Shortest path calculation (Dijkstra) is now the primary bottleneck (NEXT TARGET 🎯)

## Profiling Results Summary

### BEFORE Optimization

#### Small Graph (20 nodes, 30 edges) - 1.7s total
- `compute_derivatives`: ~0.27s in iterations

#### Medium Graph (100 nodes, 200 edges) - 4.1s total
- **`compute_derivatives`: 3.763s (92%)**
- `math.sqrt`: 1,346,400 calls, 0.087s
- `math.isfinite`: 1,346,536 calls, 0.081s
- Shortest paths: 0.158s (4%)
- Priority queue operations: ~0.15s

#### Large Graph (500 nodes, 1000 edges) - 115.8s total
- **`compute_derivatives`: 113.5s (98%)**
- `math.sqrt`: 37,450,000 calls, 2.27s
- `math.isfinite`: 37,450,136 calls, 1.986s
- Shortest paths: 10.5s (9%)

#### With Constraints (50 nodes) - 1.4s total
- **`compute_derivatives`: 1.290s (92%)**
- `math.sqrt`: 499,800 calls, 0.032s
- `math.isfinite`: 406,600 calls, 0.024s

### AFTER Optimization (Vectorization)

#### Small Graph (20 nodes, 30 edges) - 0.026s total
- **`compute_derivatives`: 0.007s (27%)** ⚡ **38x faster**
- Overall: **65x faster**

#### Medium Graph (100 nodes, 200 edges) - 0.207s total
- **`compute_derivatives`: 0.034s (16%)** ⚡ **110x faster**
- Shortest paths: 0.156s (75%) - now the bottleneck
- Priority queue operations: ~0.10s (5%)
- Overall: **19.8x faster**

#### Large Graph (500 nodes, 1000 edges) - 5.651s total
- **`compute_derivatives`: 0.682s (12%)** ⚡ **166x faster**
- Shortest paths: 4.661s (82%) - dominant bottleneck
- Priority queue operations: ~3.24s (6%)
- Overall: **20.5x faster**

#### With Constraints (50 nodes) - 0.082s total
- **`compute_derivatives`: 0.028s (34%)** ⚡ **46x faster**
- Shortest paths: 0.036s (44%)
- Overall: **17x faster**

##  Performance Hotspots

### 1. Gradient Descent - `compute_derivatives()` (descent.py:176)

**Current Implementation**:
```python
def compute_derivatives(self, x: np.ndarray) -> None:
    """Compute first and second derivatives."""
    # Nested loops: O(n²)
    for u in range(n):
        for v in range(n):
            if u == v:
                continue

            # Distance computation
            for i in range(self.k):
                dx = x[i, u] - x[i, v]  # Per-dimension
                d[i] = dx
                d2[i] = dx * dx
                distance_squared += d2[i]

            distance = math.sqrt(distance_squared)  # Called n² times

            # ... more computation per pair
```

**Issues**:
- **O(n²) complexity**: For n=500, that's 250,000 node pair iterations
- **Per-dimension loops**: Extra k loops inside n² loop
- **Repeated math calls**: `sqrt` and `isfinite` called in innermost loops
- **Scalar operations**: Not vectorized despite using NumPy

**Impact**:
- Medium graph (100 nodes): 10,000 pairs × 136 iterations = 1,360,000 pair computations
- Large graph (500 nodes): 250,000 pairs × ~150 iterations = 37,500,000 pair computations

### 2. Shortest Paths - Dijkstra (shortestpaths.py:176)

**Current Implementation**:
- Pure Python Dijkstra with pairing heap priority queue
- O(E log V) per source node, O(V × E log V) total for all-pairs

**Impact**:
- Medium graph: 0.158s (4% of total)
- Large graph: 10.5s (9% of total)
- Becomes significant only for larger graphs

### 3. Priority Queue Operations (pqueue.py)

**Current Implementation**:
- Pairing heap in pure Python
- Used heavily by Dijkstra's algorithm

**Impact**:
- Medium graph: ~0.15s across merge/remove_min/pop operations
- Not a major bottleneck compared to gradient descent

## Optimization Recommendations

### ✅ Priority 1: Vectorize `compute_derivatives` (COMPLETED)

**Status**: ✅ **IMPLEMENTED AND DEPLOYED**

**Actual Implementation**:
```python
def compute_derivatives(self, x: np.ndarray) -> None:
    """Vectorized derivative computation using NumPy broadcasting."""
    n = self.n
    if n < 1:
        return

    # Compute all pairwise differences using broadcasting
    diff = x[:, :, np.newaxis] - x[:, np.newaxis, :]  # (k, n, n)
    diff_squared = diff ** 2
    dist_squared = np.sum(diff_squared, axis=0)  # (n, n)

    # Create masks for edge cases
    diagonal_mask = np.eye(n, dtype=bool)
    distances = np.sqrt(np.maximum(dist_squared, 1e-9))

    # Vectorized weight and validity checking
    weights = self.G.copy() if self.G is not None else np.ones((n, n))
    p_stress_mask = (weights > 1) & (distances > self.D)
    weights = np.where(weights > 1, 1.0, weights)
    valid_mask = ~diagonal_mask & np.isfinite(self.D) & ~p_stress_mask

    # Vectorized gradient and Hessian computation
    # ... (full implementation in descent.py)
```

**Actual Results**:
- ✅ Small graphs: **38x faster** for `compute_derivatives`, **65x overall**
- ✅ Medium graphs: **110x faster** for `compute_derivatives`, **20x overall**
- ✅ Large graphs: **166x faster** for `compute_derivatives`, **21x overall**
- ✅ All 312 tests pass - numerical correctness maintained

**Challenges Solved**:
- ✅ Grid snap forces handled correctly
- ✅ Lock constraints work as before
- ✅ Diagonal elements properly masked
- ✅ Division by zero prevented with `np.where`
- ✅ P-stress filtering maintained

### Priority 2: Cache Distance Matrix (Medium Impact)

**Status**: ⏸️ **DEFERRED** - Not needed after vectorization

**Reason**: With vectorization, distance computation is no longer the bottleneck. The overhead of caching and cache invalidation would likely negate any benefits.

**Original Strategy**: Compute distance matrix once, reuse across iterations

**Decision**: Focus on Dijkstra optimization instead, which is now the primary bottleneck.

### 🎯 Priority 2 (NEW): Optimize Shortest Paths (NOW HIGHEST IMPACT)

**Status**: 🎯 **NEXT TARGET** - Now accounts for 75-82% of runtime

**Current Performance**:
- Medium graph (100 nodes): 0.156s (75% of 0.207s total)
- Large graph (500 nodes): 4.661s (82% of 5.651s total)

**Options**:
1. **Use scipy.sparse.csgraph.shortest_path** - C implementation, much faster (RECOMMENDED)
2. **Use NumPy for Dijkstra loops** - Vectorize priority queue operations
3. **Cache results** - Reuse if graph structure doesn't change

**Expected Speedup**: 3-5x for shortest paths
- Large graph: 5.65s → **1.5-2s total**
- Medium graph: 0.207s → **0.05-0.1s total**

**Implementation**:
```python
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# Convert edge list to sparse adjacency matrix
# Call shortest_path(adjacency, method='D', directed=False)
```

**Impact**: This would make large graphs (500 nodes) complete in **1-2 seconds** instead of 5.6s.

### Priority 3: Use Numba JIT Compilation (Medium Impact)

**Status**: 🔄 **FEASIBLE** - Could provide additional 2-5x improvement

**Strategy**: Add `@numba.jit` decorators to remaining hot functions

**Current bottlenecks** (after vectorization):
- Dijkstra's algorithm (pure Python)
- Priority queue operations

**Expected Speedup**: 2-5x for Dijkstra if scipy not used

**Trade-offs**:
- Adds dependency on numba
- scipy approach likely better for shortest paths
- Could be useful for other hot paths

**Recommendation**: Use scipy for shortest paths first, then evaluate if Numba still needed.

### Priority 4: Parallel Processing (Future Enhancement)

**Status**: ⏭️ **FUTURE** - Evaluate after scipy optimization

**Strategy**: Parallelize remaining computations

**Options**:
1. **NumPy with BLAS threading** - Already used for vectorized operations
2. **numba.prange** - For any remaining Python loops
3. **Parallel Dijkstra** - Multiple sources computed simultaneously

**Expected Speedup**: 2-4x on multi-core (diminishing returns after vectorization)

**Note**: With current vectorization + scipy, most graphs will complete in <1s, making parallelization less critical.

## Optimization Priority Matrix

### Original (Pre-Vectorization)
| Optimization | Impact | Effort | Priority | Expected Speedup |
|--------------|--------|--------|----------|------------------|
| Vectorize compute_derivatives | Very High | High | **1** | 10-50x |
| Numba JIT compilation | High | Low | **2** | 5-20x |
| Parallel processing | Medium | Medium | **3** | 4-8x |
| Cache distance matrix | Medium | Low | **4** | 2x |
| scipy shortest paths | Low | Low | **5** | 3-5x |

### Current (Post-Vectorization)
| Optimization | Impact | Effort | Priority | Status | Actual/Expected Speedup |
|--------------|--------|--------|----------|--------|------------------------|
| Vectorize compute_derivatives | Very High | High | **1** | ✅ **DONE** | **20-170x** |
| scipy shortest paths | High | Low | **2** | 🎯 **NEXT** | 3-5x |
| Numba JIT (Dijkstra) | Medium | Low | **3** | 🔄 Optional | 2-5x |
| Parallel processing | Low | Medium | **4** | ⏭️ Future | 2-4x |
| Cache distance matrix | Low | Low | **-** | ⏸️ Deferred | N/A |

## Implementation Roadmap

### ✅ Phase 1: Vectorization (COMPLETED)
1. ✅ Vectorize `compute_derivatives` with NumPy broadcasting
2. ✅ Handle edge cases (diagonal, division by zero, P-stress)
3. ✅ Maintain special features (locks, grid snap)
4. ✅ Comprehensive testing - all 312 tests pass
5. ✅ Profile and measure - achieved 20-170x speedup

**Result**: 20-65x overall speedup depending on graph size

### 🎯 Phase 2: Scipy Integration (NEXT - 1-2 days)
1. 🎯 Replace pure Python Dijkstra with `scipy.sparse.csgraph.shortest_path`
2. 🎯 Convert edge list to sparse adjacency matrix
3. 🎯 Profile and measure improvements
4. 🎯 Update tests if needed

**Expected Result**: Additional 3-5x speedup (500-node graph in ~1-2s)

### Phase 3: Optional Enhancements (Future)
1. ⏭️ Consider Numba JIT for remaining bottlenecks
2. ⏭️ Evaluate parallelization if needed
3. ⏭️ Algorithm improvements (better initialization, adaptive step sizes)
4. ⏭️ Memory optimization if needed

**Note**: After scipy integration, most use cases will have sub-second performance, making further optimization lower priority.

## Overall Performance Results

### ✅ Achieved (Vectorization Only)
- Small graphs (20 nodes): **65x faster** → **0.026s** (was 1.7s)
- Medium graphs (100 nodes): **20x faster** → **0.207s** (was 4.1s)
- Large graphs (500 nodes): **21x faster** → **5.65s** (was 115.8s)

### 🎯 Projected (After scipy Integration)
- Small graphs (20 nodes): **100-150x faster** → **~0.01-0.02s**
- Medium graphs (100 nodes): **60-100x faster** → **~0.05-0.1s**
- Large graphs (500 nodes): **60-120x faster** → **~1-2s**

### Stretch Goal (scipy + Numba + Parallelization)
- Large graph (500 nodes): **200-500x faster** → **~0.2-0.5s**
- This would make PyCola competitive with native C++ implementations

## Testing Strategy

1. **Correctness**: Verify optimized code produces identical results (within floating point tolerance)
2. **Performance**: Benchmark suite across graph sizes
3. **Regression**: Continuous profiling in CI/CD
4. **Memory**: Track memory usage to avoid excessive allocation

## Conclusion

The library's performance is dominated by gradient descent stress minimization. **Vectorizing `compute_derivatives` offers the highest ROI**, with potential 10-100x speedup. Combined with Numba JIT and parallelization, the library could achieve 100-1000x performance improvements for large graphs, making it viable for real-time interactive layouts.
