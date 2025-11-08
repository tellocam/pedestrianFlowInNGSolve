# Source Utilities for NGSolve Simulations

## Overview

This directory contains Python utility modules for pedestrian flow simulations.

## Modules

### `parameter_analysis.py`

Parameter analysis function for comprehensive numerical analysis.

**Main function:**
```python
from src import analyze_parameters

results = analyze_parameters(
    u0, rho_c, gamma_w, delta, epsilon,
    mesh, mesh_maxh, p_order, Hwid, Hcol,
    omega=None, fes_rho=None
)
```

**See documentation:** `../../docs/05_PARAMETER_ANALYSIS_TOOL.md`

## Usage in Notebooks

Since notebooks are in parent directory (`NGSolve/`):

```python
from src import analyze_parameters

results = analyze_parameters(...)
```

No need for `sys.path` manipulation - direct import works!

## Documentation

- API Reference: See docstring in `parameter_analysis.py`
- Quick Start: `../../docs/06_QUICK_START_GUIDE.md`
- Full Guide: `../../docs/05_PARAMETER_ANALYSIS_TOOL.md`
