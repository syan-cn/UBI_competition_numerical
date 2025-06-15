# Duopoly Insurance Model Framework

A framework for solving duopoly insurance models using Karush-Kuhn-Tucker (KKT) conditions.

## Overview

This framework implements a duopoly insurance model where two insurers compete by offering contracts to customers with different risk types. It uses KKT conditions to find the mathematical equilibrium.

## Features

- **KKT-Based Solver**: Mathematically rigorous solver using Karush-Kuhn-Tucker conditions
- **Flexible Functions**: Multiple functional forms for each component (linear, exponential, power, logistic)
- **Discrete State Spaces**: Support for custom probability structures
- **Visualization**: Comprehensive plotting and analysis tools

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
conda install -c conda-forge ipopt  # or: pip install cyipopt
```

2. **Run simulation:**
```bash
python duopoly.py
```

## Requirements

- Python 3.7+
- NumPy, SciPy, Matplotlib, Pandas
- Pyomo >= 6.0.0
- **Ipopt solver** (required for KKT optimization)

### Solver Dependency

This framework uses **Ipopt** (Interior Point OPTimizer) as the nonlinear optimization solver for the KKT conditions. Ipopt is essential for solving the mathematical equilibrium and must be installed separately from the Python packages.

**Installation options:**
- **Conda (recommended):** `conda install -c conda-forge ipopt`
- **Pip:** `pip install cyipopt`
- **Manual:** Download from [https://coin-or.github.io/Ipopt/](https://coin-or.github.io/Ipopt/)

## Output

- Results saved to `outputs/` directory
- Logs saved to `logs/` directory
- Debug reports in `debug_reports/` directory

## License

For research and educational purposes.
