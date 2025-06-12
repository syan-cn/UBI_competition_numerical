# Duopoly Insurance Model Framework

A flexible, modular framework for solving duopoly-style insurance models with different functional forms and parameters.

## Overview

This framework implements a duopoly insurance model where two insurers compete by offering contracts to customers with different risk types. The model includes:

- **Flexible Function Configuration**: Each function can independently use different functional forms
- **Discrete State Spaces**: Support for discrete state distributions with custom probability structures
- **Numerical Solver**: Finds optimal action schedules, premiums, and indemnity functions
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

### 5. Simulation & Analysis

The `InsuranceSimulator` class provides comprehensive analysis capabilities including full model simulation, result visualization, and parameter sensitivity analysis.

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0
- typing-extensions >= 4.0.0

## License

This framework is provided for research and educational purposes.
