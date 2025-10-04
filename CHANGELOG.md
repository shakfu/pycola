# CHANGELOG

All notable project-wide changes will be documented in this file. Note that each subproject has its own CHANGELOG.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Commons Changelog](https://common-changelog.org). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Types of Changes

- Added: for new features.
- Changed: for changes in existing functionality.
- Deprecated: for soon-to-be removed features.
- Removed: for now removed features.
- Fixed: for any bug fixes.
- Security: in case of vulnerabilities.

---

## [0.1.1] - Performance Optimization Release

### Added
- Comprehensive performance profiling system (`scripts/profile_layout.py`)
- Performance analysis documentation (`docs/OPTIMIZATION_ANALYSIS.md`)
- Performance comparison documentation (`docs/PERFORMANCE_COMPARISON.md`)
- Performance benchmarks in README.md and CLAUDE.md

### Changed
- **MAJOR PERFORMANCE IMPROVEMENT**: Vectorized `compute_derivatives()` in `descent.py` using NumPy broadcasting
  - **20-65x overall speedup** depending on graph size
  - **110-170x faster** for gradient descent computation specifically
  - Medium graphs (100 nodes): 4.1s → 0.2s (20x faster)
  - Large graphs (500 nodes): 115.8s → 5.6s (21x faster)
- Replaced nested Python loops with NumPy array operations in gradient descent
- All edge cases properly handled (diagonal elements, division by zero, P-stress filtering)
- Updated Makefile to use `uv` for dependency management
- Added `from __future__ import annotations` to all modules for forward type references
- Updated CLAUDE.md with current performance metrics and optimization roadmap

### Fixed
- Forward reference type hints in `vpsc.py`, `powergraph.py`, `descent.py`, and `layout.py`
- Import paths in profiling scripts
- Source directory path corrections in Makefile (`src/pycola` vs `pycola`)

### Performance
- **Small graphs (20 nodes)**: ~0.03s (was ~1.7s) - **65x faster**
- **Medium graphs (100 nodes)**: ~0.2s (was ~4.1s) - **20x faster**
- **Large graphs (500 nodes)**: ~5.6s (was ~115.8s) - **21x faster**
- New bottleneck identified: Shortest path calculation (Dijkstra) - 75-82% of runtime
- Next optimization target: Replace Dijkstra with scipy for potential 3-5x additional improvement

### Testing
- All 312 tests pass with vectorized implementation
- Numerical correctness maintained (floating-point accuracy within tolerance)
- Test suite completes in 0.41s

## [0.1.0] - Initial Release

### Added
- Complete Python port of WebCola graph layout library
- 2D force-directed layout with gradient descent
- 3D layout support
- VPSC (Variable Placement with Separation Constraints) solver
- Constraint-based layout (separation, alignment)
- Overlap avoidance with rectangle projection
- Hierarchical group layouts with containment
- Power graph automatic clustering
- Grid router for orthogonal edge routing
- Event system (start/tick/end events)
- Fluent API with method chaining
- Disconnected component handling
- Link length calculators (symmetric difference, Jaccard)
- Flow layouts (directed graph layouts)
- Interactive drag support
- Comprehensive test suite (312 tests, 100% pass rate)
- Priority queue (pairing heap) implementation
- Red-black tree implementation
- Shortest paths (Dijkstra) implementation
- Computational geometry utilities
- Batch layout operations
- Complete documentation with examples
- CLAUDE.md with architecture overview
- TypeScript to Python translation guide

### Dependencies
- numpy>=1.20.0
- sortedcontainers>=2.4.0

### Development
- pytest>=8.3.5
- pytest-cov>=5.0.0
- mypy>=1.14.1
- ruff>=0.13.3
- Uses `uv` for dependency management
- Python 3.9+ required
- MIT License
