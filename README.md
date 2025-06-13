# Duopoly Insurance Model Framework

A flexible, modular framework for solving duopoly-style insurance models with different functional forms and parameters.

## Overview

This framework implements a duopoly insurance model where two insurers compete by offering contracts to customers with different risk types. The model includes:

- **Flexible Function Configuration**: Each function can independently use different functional forms
- **Discrete State Spaces**: Support for discrete state distributions with custom probability structures
- **Numerical Solver**: Finds optimal action schedules, premiums, and indemnity functions
- **Parallel Processing**: High-performance parallel grid search for computational efficiency
- **Simulation & Plotting**: Visualizes results and performs sensitivity analysis

## Framework Components

### 1. Discrete State Spaces

The framework supports discrete state spaces through the `DiscreteStateSpace` class, allowing for custom probability structures and binary state distributions.

### 2. Available Functional Forms

The framework supports multiple functional forms for each component.

### 3. Flexible Function Configuration

Configure different functional forms for each component independently, allowing for mixed functional form combinations.

### 4. Numerical Solver

The `DuopolySolver` class implements the numerical solution of the duopoly model, handling:

- Reservation utility computation
- Incentive constraint solving
- Optimal contract determination
- Grid-based discretization
- **Parallel grid search**: Multi-core processing for large-scale computations

### 5. Parallel Processing

The framework includes high-performance parallel processing capabilities:

- **Parallel Contract Generation**: Distributes contract feasibility evaluation across CPU cores
- **Parallel Contract Pair Evaluation**: Parallelizes contract pair evaluation and profit computation
- **Automatic CPU Detection**: Uses all available cores by default
- **Configurable Workers**: Specify number of parallel workers as needed

#### Usage Examples

```python
# Basic parallel processing
pareto_solutions = solver.brute_force_duopoly(n_jobs=4)

# Using the main simulation function with parallel processing
results = run_simulation(
    state_spaces=[{'name': 'binary', 'f': 'binary_states'}],
    params=params,
    n_jobs=4  # Use 4 CPU cores
)

# Use all available cores (default)
results = run_simulation(
    state_spaces=[{'name': 'binary', 'f': 'binary_states'}],
    params=params
)
```

#### Performance Characteristics

- **Small grids** (10×5): 2-4x speedup
- **Medium grids** (20×10): 4-8x speedup  
- **Large grids** (50×20): 8-16x speedup

### 6. Simulation & Analysis

The `InsuranceSimulator` class provides comprehensive analysis capabilities including full model simulation, result visualization, and parameter sensitivity analysis.

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0
- typing-extensions >= 4.0.0

## License

This framework is provided for research and educational purposes.
