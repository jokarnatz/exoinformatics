# exoinformatics

**Author:** David Kipping, Columbia University  
**Reference:** "Do planets remember how they formed?", MNRAS ([arXiv](https://arxiv.org/))

Exoplanet informatics utilities for computing planetary entropy microstates using three methods: **tally**, **integral**, and **change-point**.

This is a refactored Python 3 version of the original [CoolWorlds/exoinformatics](https://github.com/CoolWorlds/exoinformatics) code. Changes include Python 3 compatibility, type hints, docstrings, `math.comb` in place of `scipy.special.binom`, a bug fix in `findk` (now raises `ValueError` instead of an unbound-name crash when no microstate matches), and a more idiomatic `randomswap` implementation.

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
import exoinformatics

radii = np.sort(np.random.rand(5))   # rank-ordered planetary radii
N = len(radii)

# Tally score and microstate index
tally = [2 * exoinformatics.heaviside(radii[i+1] - radii[i]) - 1 for i in range(N-1)]
T = int(sum(tally))
m0 = exoinformatics.M0(N, T)

# Tally microstate occupancy
W_T = exoinformatics.W0(N, m0)

# Integral score and sub-microstate index
tally_history = [0] * N
for i in range(N-1):
    tally_history[i+1] = int(sum(tally[:i+1]))
area = sum(0.5 * (tally_history[i+1] + tally_history[i]) for i in range(N-1))
k = exoinformatics.findk(N, m0, area)
W_I = exoinformatics.WI(N, m0, k)

# Change-point score
delta = [1 - exoinformatics.kronecker(tally[i], tally[i+1]) for i in range(N-2)]
D = int(sum(delta))
W_C = exoinformatics.WC(N, m0, D)

print(f"W_T={W_T}, W_I={W_I}, W_C={W_C}")
```

Run the bundled example:

```bash
python test.py
```

## API

| Function | Description |
|---|---|
| `heaviside(x)` | Heaviside step function |
| `kronecker(x, y)` | Kronecker delta |
| `eulerian(n, q)` | Eulerian number ⟨n, q⟩ |
| `M0(N, T)` | Tally microstate index |
| `W0(N, M)` | Tally microstate occupancy |
| `Omega0(N)` | Number of tally macrostates |
| `omegai(N, M)` | Number of integral sub-microstates |
| `OmegaI(N)` | Total integral microstates |
| `findk(N, M, I)` | Integral sub-microstate index |
| `WI(N, M, k)` | Integral microstate occupancy |
| `omegac(N, M)` | Number of change-point sub-microstates |
| `OmegaC(N)` | Total change-point microstates |
| `WC(N, M, D)` | Change-point microstate occupancy |
| `randomswap(radii)` | Swap a random adjacent pair in a radii list |

> **Note:** The lookup tables in `WI` and `WC` currently support N ≤ 10. Inputs outside this range return `-1`.

## Visualizations

The PDF files in the original repository depict the (sub-)microstates of each entropy method. The CDF files are Mathematica CDF Player graphics showing 3D microstate visualizations.
