# Duopoly Insurance Model Framework

A flexible, modular framework for solving duopoly-style insurance models with different functional forms and parameters.

## Overview

This framework implements a duopoly insurance model where two insurers compete by offering contracts to customers with different risk types. The model includes:

- **Flexible Function Configuration**: Each function can independently use different functional forms
- **Discrete State Spaces**: Support for discrete state distributions with custom probability structures
- **Numerical Solver**: Finds optimal action schedules, premiums, and indemnity functions
- **Advanced Parallel Processing**: High-performance parallel grid search with intelligent evaluation strategies
- **Efficient Contract Pair Evaluation**: Multiple approaches to evaluate contract pairs after generation
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
- **Advanced parallel processing**: Multi-core processing with intelligent evaluation strategies

### 5. Contract Pair Evaluation Methods

The framework provides multiple optimized approaches to evaluate contract pairs after obtaining all feasible contracts:

#### 5.1 Original Method
- **Description**: Evaluates all contract pairs at once using parallel processing

#### 5.2 Simple Efficient Method
- **Description**: Pre-filters contracts using heuristics, then uses optimized batching
- **Features**:
  - Pre-filtering removes clearly dominated contracts (low profit, extreme premiums)
  - Optimized batch sizes for parallel processing
  - Progress tracking during evaluation
  - Adaptive batch sizing based on available resources

#### 5.3 Incremental Pareto Method
- **Description**: Builds Pareto frontier incrementally by processing contracts in chunks
- **Features**:
  - Processes contracts in manageable chunks (default: 500x500)
  - Maintains running Pareto frontier throughout process
  - Memory efficient for very large problems
  - Configurable chunk sizes

#### 5.4 Divide-and-Conquer Method
- **Description**: Recursively partitions contract pairs into subsets and evaluates in parallel
- **Features**:
  - Recursive subdivision of contract space
  - Parallel evaluation of quadrants
  - Efficient Pareto frontier merging
  - Configurable recursion depth

### 6. Traditional Parallel Processing

- **Parallel Contract Generation**: Distributes contract feasibility evaluation across CPU cores
- **Parallel Contract Pair Evaluation**: Parallelizes contract pair evaluation and profit computation
- **Automatic CPU Detection**: Uses all available cores by default
- **Configurable Workers**: Specify number of parallel workers as needed

### 7. Simulation & Analysis

The `InsuranceSimulator` class provides comprehensive analysis capabilities including full model simulation, result visualization, and parameter sensitivity analysis.

## Demonstration

Run the demonstration script to see the different evaluation methods:

```bash
python evaluation_methods_demo.py
```

This script compares all four evaluation methods and generates comprehensive performance comparison plots including:
- Execution time comparisons across problem sizes
- Solution quality verification (all methods produce identical results)
- Speedup analysis and scalability characteristics
- Memory usage patterns and recommendations

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0
- typing-extensions >= 4.0.0

## License

This framework is provided for research and educational purposes.
