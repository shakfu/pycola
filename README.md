# PyCola - Python Graph Layout Library

PyCola is a Python port of [WebCola](https://github.com/tgdwyer/WebCola), a constraint-based graph layout library, which itself it a Javascript port of `libcola` from the [adaptagrams library](https://www.adaptagrams.org). It provides force-directed layout algorithms with support for constraints, groups, and overlap avoidance.

This is my second port of an adaptagrams library, the first being [pyhola](https://github.com/shakfu/pyhola), which is a pybind11 wrapper for the adaptagrams HOLA graph layout algorithm.

You can see examples of graphs which use this layout engine on [cola.js](https://ialab.it.monash.edu/webcola/).

## Features

- **Force-Directed Layout** - 2D and 3D graph layouts using gradient descent
- **Constraint-Based** - Separation, alignment, and custom constraints
- **Overlap Avoidance** - Non-overlapping node placement
- **Hierarchical Groups** - Nested group layouts with containment
- **Power Graph** - Automatic hierarchical clustering
- **Grid Router** - Orthogonal edge routing
- **Event System** - Animation support with tick events
- **Fluent API** - Method chaining for easy configuration

## Installation

```bash
uv pip install pycola
```

## Development Setup

```bash
# Quick setup (recommended)
make sync

# Or using uv directly
uv sync
```

## Development Commands

**Using Make (recommended):**
```bash
make help          # Show all commands
make test          # Run tests
make test-coverage # Run with coverage
make typecheck     # Type checking with mypy
make format        # Format code
make lint          # Lint code
make verify        # Run all checks
make fix           # Auto-fix issues
make clean         # Clean artifacts
```

**Direct commands:**
```bash
uv run pytest                                     # Run tests
uv run pytest --cov=src/pycola --cov-report=html  # Coverage
uv run mypy src/pycola                            # Type check
uv run ruff format src/pycola tests               # Format
uv run ruff check src/pycola tests                # Lint
```

## Quick Start

### Basic Usage

```python
from pycola.layout import Layout

# Create your graph
nodes = [
    {'x': 0, 'y': 0},           # node 0
    {'x': 100, 'y': 0},         # node 1
    {'x': 200, 'y': 0},         # node 2
]

edges = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
    {'source': 2, 'target': 0},
]

# Create and configure layout
layout = Layout()
layout.nodes(nodes)
layout.links(edges)
layout.start()

# After layout, nodes have computed x, y positions
print(f"Node 0: ({nodes[0]['x']:.2f}, {nodes[0]['y']:.2f})")
print(f"Node 1: ({nodes[1]['x']:.2f}, {nodes[1]['y']:.2f})")
print(f"Node 2: ({nodes[2]['x']:.2f}, {nodes[2]['y']:.2f})")
```

### Fluent API (Method Chaining)

```python
from pycola.layout import Layout

nodes = [{'x': 0, 'y': 0} for _ in range(5)]
edges = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
    {'source': 2, 'target': 3},
    {'source': 3, 'target': 4},
]

# Configure everything with method chaining
layout = (Layout()
    .size([800, 600])              # canvas size
    .nodes(nodes)
    .links(edges)
    .link_distance(100)            # desired edge length
    .convergence_threshold(0.01)   # when to stop
    .start(50)                     # run 50 iterations
)

# Nodes now have their positions
for i, node in enumerate(nodes):
    print(f"Node {i}: ({node['x']:.1f}, {node['y']:.1f})")
```

## Advanced Features

### Overlap Avoidance

Prevent nodes from overlapping by specifying widths and heights:

```python
from pycola.layout import Layout

nodes = [
    {'x': 0, 'y': 0, 'width': 50, 'height': 30},
    {'x': 50, 'y': 0, 'width': 50, 'height': 30},
    {'x': 100, 'y': 0, 'width': 50, 'height': 30},
]

edges = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
]

layout = (Layout()
    .nodes(nodes)
    .links(edges)
    .avoid_overlaps(True)    # prevent node overlaps
    .start()
)
```

### Event-Driven Layout (Animation)

Listen to layout events for animation:

```python
from pycola.layout import Layout, EventType

nodes = [{'x': 0, 'y': 0} for _ in range(10)]
edges = [{'source': i, 'target': i+1} for i in range(9)]

layout = Layout().nodes(nodes).links(edges)

# Listen to layout events
def on_tick(event):
    print(f"Tick {event['alpha']:.3f}, stress: {event['stress']:.2f}")

def on_end(event):
    print("Layout complete!")

layout.on(EventType.tick, on_tick)
layout.on(EventType.end, on_end)

layout.start(100)  # 100 iterations
```

### Hierarchical Groups

Create nested group layouts:

```python
from pycola.layout import Layout, Group

# Nodes
nodes = [
    {'x': 0, 'y': 0, 'width': 30, 'height': 30},    # 0
    {'x': 50, 'y': 0, 'width': 30, 'height': 30},   # 1
    {'x': 100, 'y': 0, 'width': 30, 'height': 30},  # 2
    {'x': 150, 'y': 0, 'width': 30, 'height': 30},  # 3
]

# Edges
edges = [
    {'source': 0, 'target': 1},
    {'source': 2, 'target': 3},
    {'source': 1, 'target': 2},
]

# Groups (clusters)
groups = [
    Group(leaves=[nodes[0], nodes[1]], padding=10),  # group 1
    Group(leaves=[nodes[2], nodes[3]], padding=10),  # group 2
]

# Layout with groups
layout = (Layout()
    .size([400, 300])
    .nodes(nodes)
    .links(edges)
    .groups(groups)
    .avoid_overlaps(True)
    .link_distance(80)
    .start()
)

# Access results
for i, node in enumerate(nodes):
    print(f"Node {i}: ({node['x']:.1f}, {node['y']:.1f})")
```

### Fixed Nodes

Keep specific nodes at fixed positions:

```python
nodes = [
    {'x': 0, 'y': 0, 'fixed': 3},     # fixed at (0, 0)
    {'x': 100, 'y': 0, 'fixed': 1},   # x fixed, y free
    {'x': 200, 'y': 0},                # completely free
]

edges = [
    {'source': 0, 'target': 1},
    {'source': 1, 'target': 2},
]

layout = Layout().nodes(nodes).links(edges).start()
```

### 3D Layout

```python
from pycola.layout3d import Layout3D, Node3D, Link3D

nodes = [
    Node3D(0, 0, 0),
    Node3D(1, 0, 0),
    Node3D(0, 1, 0),
]

links = [
    Link3D(0, 1),
    Link3D(1, 2),
    Link3D(2, 0),
]

layout = Layout3D(nodes, links, ideal_link_length=1.0)
layout.start(iterations=100)

# Access 3D positions
for i, node in enumerate(nodes):
    print(f"Node {i}: ({node.x:.2f}, {node.y:.2f}, {node.z:.2f})")
```

### Power Graph (Hierarchical Clustering)

Automatically detect and cluster graph structure:

```python
from pycola.layout import Layout

nodes = [{'x': 0, 'y': 0} for _ in range(6)]
edges = [
    {'source': 0, 'target': 1},
    {'source': 0, 'target': 2},
    {'source': 3, 'target': 4},
    {'source': 3, 'target': 5},
]

def on_power_graph(power_graph):
    print(f"Found {len(power_graph.groups)} groups")
    for group in power_graph.groups:
        print(f"  Group with {len(group.leaves)} nodes")

layout = (Layout()
    .nodes(nodes)
    .links(edges)
    .power_graph_groups(on_power_graph)
    .start()
)
```

## API Reference

### Layout Class

The main entry point for graph layout.

#### Configuration Methods

| Method | Description |
|--------|-------------|
| `nodes(list)` | Set the graph nodes |
| `links(list)` | Set the graph edges |
| `size([w, h])` | Set canvas dimensions |
| `link_distance(n)` | Set desired edge length (number or function) |
| `avoid_overlaps(bool)` | Enable/disable overlap avoidance |
| `handle_disconnected(bool)` | Enable/disable disconnected component handling |
| `groups(list)` | Set hierarchical groups |
| `constraints(list)` | Set layout constraints |
| `flow_layout(axis, gap)` | Enable flow layout (x or y axis) |
| `convergence_threshold(n)` | Set convergence threshold (default 0.01) |
| `default_node_size(n)` | Set default node size |
| `group_compactness(n)` | Set group compactness (0-1) |

#### Link Length Calculators

| Method | Description |
|--------|-------------|
| `symmetric_diff_link_lengths(w)` | Use symmetric difference for link lengths |
| `jaccard_link_lengths(w)` | Use Jaccard similarity for link lengths |
| `link_type(accessor)` | Set link type accessor |

#### Layout Control

| Method | Description |
|--------|-------------|
| `start(iterations, init_temp, cool_rate, cool_step, center)` | Start the layout |
| `tick()` | Run one iteration |
| `stop()` | Stop the layout |
| `resume()` | Resume a stopped layout |
| `alpha()` | Get current alpha (temperature) |

#### Event System

| Method | Description |
|--------|-------------|
| `on(event, callback)` | Register event listener |
| `trigger(event)` | Trigger an event |

Available events:
- `EventType.start` - Layout started
- `EventType.tick` - Each iteration
- `EventType.end` - Layout completed

#### Interactive Methods

| Method | Description |
|--------|-------------|
| `drag_start(node)` | Start dragging a node |
| `drag(node, position)` | Drag node to position |
| `drag_end(node)` | End dragging |
| `mouse_over(node)` | Handle mouse over |
| `mouse_out(node)` | Handle mouse out |

### Node Format

Nodes can be dictionaries or objects with these properties:

```python
{
    'x': 0,           # initial x (optional, will be randomized)
    'y': 0,           # initial y (optional, will be randomized)
    'width': 50,      # node width (for overlap avoidance)
    'height': 30,     # node height (for overlap avoidance)
    'fixed': 0,       # 0=free, 1=fixed x, 2=fixed y, 3=fixed both
    # ... any custom properties
}
```

### Edge/Link Format

Edges can be dictionaries or objects:

```python
{
    'source': 0,      # source node index or reference
    'target': 1,      # target node index or reference
    'length': 100,    # desired length (optional)
    'weight': 1.0,    # importance (0-1, optional)
}
```

### Group Format

```python
from pycola.layout import Group

group = Group(
    leaves=[node1, node2],           # leaf nodes
    groups=[subgroup1, subgroup2],   # nested groups
    padding=10                        # padding around group
)
```

### Constraints

```python
from pycola.linklengths import SeparationConstraint, AlignmentConstraint

# Separation constraint
sep = SeparationConstraint(
    axis='x',        # 'x' or 'y'
    left=0,          # left node index
    right=1,         # right node index
    gap=50,          # minimum gap
    equality=False   # if True, gap is exact
)

# Alignment constraint
align = AlignmentConstraint(
    axis='y',
    offsets=[
        {'node': 0, 'offset': 0},
        {'node': 1, 'offset': 0},
        {'node': 2, 'offset': 0},
    ]
)

layout.constraints([sep, align])
```

## Module Overview

- **`pycola.layout`** - Main 2D force-directed layout
- **`pycola.layout3d`** - 3D force-directed layout
- **`pycola.descent`** - Gradient descent optimizer
- **`pycola.vpsc`** - Variable Placement with Separation Constraints solver
- **`pycola.rectangle`** - Rectangle operations and overlap removal
- **`pycola.geom`** - Computational geometry utilities
- **`pycola.powergraph`** - Hierarchical graph clustering
- **`pycola.gridrouter`** - Orthogonal edge routing
- **`pycola.shortestpaths`** - Shortest path algorithms
- **`pycola.linklengths`** - Link length utilities and constraints
- **`pycola.handledisconnected`** - Disconnected component handling
- **`pycola.batch`** - Batch layout operations
- **`pycola.pqueue`** - Priority queue implementation
- **`pycola.rbtree`** - Red-black tree implementation

## Examples

### Simple Network

```python
from pycola.layout import Layout

# Create a simple network
nodes = [{'x': 0, 'y': 0} for _ in range(5)]
edges = [
    {'source': 0, 'target': 1},
    {'source': 0, 'target': 2},
    {'source': 1, 'target': 3},
    {'source': 2, 'target': 3},
    {'source': 3, 'target': 4},
]

layout = (Layout()
    .size([400, 300])
    .nodes(nodes)
    .links(edges)
    .link_distance(80)
    .start(100)
)

for i, node in enumerate(nodes):
    print(f"Node {i}: ({node['x']:.1f}, {node['y']:.1f})")
```

### Tree Layout

```python
from pycola.layout import Layout

# Create tree structure
nodes = [{'x': 0, 'y': 0} for _ in range(7)]
edges = [
    {'source': 0, 'target': 1},  # root to children
    {'source': 0, 'target': 2},
    {'source': 1, 'target': 3},  # level 2
    {'source': 1, 'target': 4},
    {'source': 2, 'target': 5},
    {'source': 2, 'target': 6},
]

layout = (Layout()
    .nodes(nodes)
    .links(edges)
    .flow_layout('y', 80)  # vertical flow
    .start()
)
```

### Custom Link Distances

```python
from pycola.layout import Layout

nodes = [{'x': 0, 'y': 0} for _ in range(4)]

# Custom link distances
edges = [
    {'source': 0, 'target': 1, 'length': 50},
    {'source': 1, 'target': 2, 'length': 100},
    {'source': 2, 'target': 3, 'length': 150},
]

layout = Layout().nodes(nodes).links(edges).start()
```

## Translation Status

[x] **Complete** - All core modules translated and tested

- [x] Core data structures (pqueue, types)
- [x] VPSC constraint solver
- [x] Gradient descent optimization
- [x] Geometry utilities
- [x] Rectangle and projection
- [x] Grid router
- [x] Power graph
- [x] Main layout engine (2D and 3D)
- [x] Batch operations
- [x] Comprehensive test suite (312 tests, 100% passing)

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_layout.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pycola
```

Test statistics:
- **312 tests** covering all modules
- **100% pass rate**
- Comprehensive coverage of algorithms and edge cases

## Performance

**Current performance** (after NumPy vectorization optimization):
- Small graphs (20 nodes): ~0.03s
- Medium graphs (100 nodes): ~0.2s
- Large graphs (500 nodes): ~5.6s

**Optimizations**:
- Vectorized gradient descent with NumPy broadcasting (20-170x faster)
- Efficient matrix operations for O(n²) computations
- Runge-Kutta integration for gradient descent
- Spatial indexing with red-black trees
- Optimized VPSC constraint solver

See `docs/PERFORMANCE_COMPARISON.md` for detailed benchmarks.

## Architecture

See [../CLAUDE.md](../CLAUDE.md) for overall architecture documentation.

The Python implementation follows the same structure as the TypeScript version:

- `vpsc.py` - Variable Placement with Separation Constraints solver
- `descent.py` - Gradient descent stress minimization
- `geom.py` - Geometric primitives and algorithms
- `rectangle.py` - Rectangle overlap removal and projection
- `gridrouter.py` - Orthogonal edge routing
- `shortestpaths.py` - All-pairs shortest paths (Dijkstra)
- `powergraph.py` - Hierarchical graph clustering
- `layout.py` - Main layout orchestration
- `layout3d.py` - 3D layout engine
- `pqueue.py` - Priority queue (pairing heap)

## Dependencies

**Runtime:**
- `numpy` - Matrix operations for gradient descent
- `sortedcontainers` - Sorted data structures for sweep algorithms

**Development:**
- `pytest` - Testing framework
- `mypy` - Static type checking
- `ruff` - Fast Python linter

## Credits

PyCola is a Python port of [WebCola](https://github.com/tgdwyer/WebCola) by Tim Dwyer.

Original WebCola paper:
> Tim Dwyer, Kim Marriott, and Michael Wybrow. 2009.
> "Dunnart: A constraint-based network diagram authoring tool."
> In Graph Drawing, pages 420-431. Springer.

## License

MIT (same as original WebCola)
