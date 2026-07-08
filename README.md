# zifor — Isolation Forest with Missing Values

[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**zifor** is a Python implementation of the Isolation Forest algorithm for anomaly detection, with native support for **missing (masked) values**. The core tree construction and scoring are implemented in C++20 via nanobind for performance.

## Installation

```bash
pip install .
```

For development (with pytest and nanobind):

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Algorithm Overview

### Standard Isolation Forest

The classic Isolation Forest (Liu, Ting & Zhou, 2008) builds an ensemble of binary trees by recursively selecting a random feature and a random split value uniformly between the feature's min and max in the current subsample. Anomaly score is derived from the average path length to isolate a sample: shorter paths indicate anomalies.

### zifor: Handling Missing Values

zifor extends Isolation Forest to handle **missing (masked) values** in the data. The input is a NumPy masked array (`numpy.ma.MaskedArray`), where $\mathit{mask}[i, j] = \mathit{True}$ means the value at position $(i, j)$ is missing.

#### Split Selection

When selecting a split at a node, only **observed (non-masked)** values contribute to the feature's value range. For a set of objects $S$ at a node, for each feature $j$ we compute:

$$
\begin{aligned}
\mathit{count}_j &= |\{ i \in S \mid \mathit{mask}[i, j] = \mathit{False} \}| \\
\mathit{min}_j &= \min\{ X[i, j] \mid i \in S, \mathit{mask}[i, j] = \mathit{False} \} \\
\mathit{max}_j &= \max\{ X[i, j] \mid i \in S, \mathit{mask}[i, j] = \mathit{False} \}
\end{aligned}
$$

A feature is eligible for splitting only if $\mathit{count}_j > 1$ (at least two observed values). Among eligible features, one is chosen uniformly at random from those with maximal $\mathit{count}_j$. The split value is then drawn uniformly:

$$
v \sim U(\mathit{min}_j,\ \mathit{max}_j)
$$

#### Splitting with Missing Values

When a split on feature $j$ with threshold $v$ is applied, each object $i$ follows one of three rules:

- If $\mathit{mask}[i, j] = \mathit{True}$ (value missing): the object goes to **both** left and right children.
- If $X[i, j] < v$: goes to the **left** child.
- Otherwise: goes to the **right** child.

This "both branches" rule is the key difference from standard Isolation Forest.

#### Leaf Density via Iterative Refinement

Because missing values cause objects to appear in multiple leaves, we compute a **leaf density** vector $\tau$ (one entry per leaf) via an iterative fixed-point algorithm (up to `max_iter` iterations).

Let $L$ be the number of leaves, $N$ the number of training objects, and $\mathit{index}(i)$ the set of leaves that object $i$ reaches. Initialize:

$$
\tau_\ell^{(0)} = \frac{1}{L} \quad \forall \ell
$$

At each iteration $t$:

1. **Unroll** — for each object $i$, normalize the current $\tau$ over the leaves it belongs to:

$$
T_{i,\ell} = \frac{\tau_\ell^{(t)}}{\sum_{\ell' \in \mathit{index}(i)} \tau_{\ell'}^{(t)}} \quad \text{if } \ell \in \mathit{index}(i), \text{ else } 0
$$

2. **Collect** — sum contributions per leaf:

$$
\tau_\ell^{(t+1)} = \sum_{i: \ell \in \mathit{index}(i)} T_{i,\ell}
$$

3. **Normalize**:

$$
\tau_\ell^{(t+1)} \gets \frac{\tau_\ell^{(t+1)}}{\sum_{\ell'} \tau_{\ell'}^{(t+1)}}
$$

This converges to a distribution $\tau_\ell$ representing the probability that a randomly chosen object "lands" in leaf $\ell$, accounting for the branching ambiguity caused by missing values.

#### Anomaly Score

For a new sample $x$, the anomaly score is the weighted average of path lengths across all leaves it reaches:

$$
s(x) = \frac{\sum_{\ell \in \mathit{leaves}(x)} \frac{\tau_\ell}{|\ell|} \cdot d_\ell}{\sum_{\ell \in \mathit{leaves}(x)} \frac{\tau_\ell}{|\ell|}}
$$

where $|\ell|$ is the number of training objects in leaf $\ell$, and $d_\ell$ is the path depth adjusted for leaf size:

$$
d_\ell = \begin{cases}
\mathit{depth}(\ell) & \text{if } |\ell| = 1 \\[4pt]
\mathit{depth}(\ell) + 2\bigl(\gamma - 1 + \log|\ell|\bigr) & \text{otherwise}
\end{cases}
$$

with $\gamma \approx 0.5772$ being the Euler–Mascheroni constant. This adjustment mirrors the standard Isolation Forest's average path length for unsuccessful searches in a BST.

The final score across the ensemble is the mean of $s(x)$ over all trees.

## Project Structure

```
├── CMakeLists.txt          # CMake build for C++ extension
├── pyproject.toml          # Python project metadata
├── src/
│   └── zifor/
│       ├── __init__.py     # Package init
│       ├── forest.py       # Python ensemble wrapper
│       └── tree.cpp        # C++ tree implementation (nanobind)
└── tests/
    └── test_tree.py        # Unit tests
```

## Dependencies

- Python $\ge$ 3.9
- numpy
- nanobind (build-time)
- scikit-build-core (build-time)

## License

GNU General Public License v3 or later (GPL-3.0-or-later).
