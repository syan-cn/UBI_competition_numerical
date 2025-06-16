# Duopoly Insurance Model Framework

A framework for solving duopoly insurance models using Karush-Kuhn-Tucker (KKT) conditions.

## Overview

This framework implements a duopoly insurance model where two insurers compete by offering contracts to customers with different risk types. It uses KKT conditions to find mathematical equilibrium solutions.

## Features

- **KKT-Based Solver**: Mathematically rigorous equilibrium computation
- **Flexible Functions**: Multiple functional forms (linear, exponential, power, logistic)
- **Multi-Solver Support**: Integration with Ipopt, Knitro, and other solvers
- **Comprehensive Analysis**: Built-in visualization and logging tools
- **Modular Design**: Clean, maintainable code structure

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install optimization solver:**
```bash
# Option A: Ipopt (recommended)
conda install -c conda-forge ipopt
# OR
pip install cyipopt

# Option B: Knitro (commercial)
# Download from https://www.artelys.com/knitro
```

## Project Structure

```
├── main.py                 # Main entry point
├── elements/               # Core model components
├── solver/                 # Optimization logic
├── utils/                  # Utilities and tools
├── outputs/                # Simulation results
└── logs/                   # Execution logs
```

## Requirements

- Python 3.7+
- NumPy, SciPy, Matplotlib, Pandas, Pyomo
- **Optimization Solver**: Ipopt or Knitro (required for KKT optimization)

## Output

- Results saved to `outputs/` directory
- Logs saved to `logs/` directory
- Debug reports in `debug_reports/` directory

## License

For research and educational purposes.
