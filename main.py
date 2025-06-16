"""
Main Module for Duopoly Insurance Model

This module provides the main entry point and example usage for the modularized
duopoly insurance model framework.
"""

from elements.function_set import Functions
from solver.helper import DuopolySolver
from utils.logger import SimulationLogger


def main():
    """Main function demonstrating the duopoly insurance model."""
    
    # Example parameters
    params = {
        'W': 500.0,            # Initial wealth
        's': 200.0,             # Accident severity
        'N': 100,              # Number of customers
        'delta1': 0.9,         # Insurer 1 monitoring level
        'delta2': 0.2,         # Insurer 2 monitoring level
        'theta_min': 0.05,      # Minimum risk type
        'theta_max': 1,      # Maximum risk type
        'n_theta': 10,          # Number of risk types
        'a_min': 0.05,          # Minimum action level
        'a_max': 1.0,          # Maximum action level
        'mu': 2.0,             # Logit model scale parameter
        'p_alpha': 0.0,        # No accident probability parameter
        'p_beta': 1.0,         # No accident probability parameter
        'e_kappa': 15.0,       # Action cost parameter
        'e_power': 2.0,        # Action cost power
        'f_p_base': 0.5,       # State density parameter
        'c_lambda': 10.0,       # Insurer cost parameter
        'm_gamma': 20.0,        # Monitoring cost parameter
        'u_rho': 1e-3,          # Utility parameter
        'u_max_val': 100.0,        # Utility parameter
    }
    
    print("="*60)
    print("DUOPOLY INSURANCE MODEL DEMONSTRATION")
    print("="*60)

    # Create logger for analysis
    logger = SimulationLogger(
        experiment_name="duopoly_demo",
        log_level="INFO"
    )
    
    # Create function configuration
    function_config = {
        'p': 'linear',
        'm': 'linear',
        'e': 'power',
        'u': 'exponential',
        'f': 'binary_states',
        'c': 'linear'
    }
    
    # Create function set
    functions = Functions(function_config)
    
    # Run KKT-based simulation
    print("\n" + "="*40)
    print("KKT-BASED SOLVER")
    print("="*40)
    
    solver = DuopolySolver(functions, params)
    
    # Run simulation using the simplified run method
    success, solution = solver.run(
        solver_name='knitroampl',
        verbose=False,
        save_plots=True,
        logger=logger,
        executable_path='/Users/syan/knitro-14.2.0-ARM-MacOS/knitroampl/knitroampl'
    )
    
    if success:
        print("✅ KKT-based simulation completed!")
        print(f"Solve time: {solution.get('solve_time', 'N/A')} seconds")
    else:
        print("❌ KKT-based simulation failed!")


if __name__ == "__main__":
    main() 