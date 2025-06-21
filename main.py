"""
Main Module for Duopoly Insurance Model

This module provides the main entry point and example usage for the modularized
duopoly insurance model framework.
"""

from elements.function_set import Functions
from solver.kkt import DuopolySolverKKT
from utils.logger import SimulationLogger


def main():
    """
    Main function demonstrating the duopoly insurance model.
    
    """
    
    # OVERFLOW-SAFE parameters designed to work with tight variable bounds
    params = {
        'W': 1000.0,  # $1,000 wealth
        's': 500.0,
        'N': 20,  # 20 customers (small market)
        'delta1': 0.4,  # 60% monitoring efficiency (high-tech insurer)
        'delta2': 0.2,  # 40% monitoring efficiency (traditional insurer)
        'theta_min': 0.05,
        'theta_max': 1,
        'n_theta': 5,  # risk types
        'mu': 50,  # Moderate choice sensitivity
        'p_alpha': 0,  # Base accident probability scaling
        'p_beta': 1,  # Base accident probability scaling
        'p_hat': 0.02,
        'n_trials': 4,
        'e_kappa': 100.0,  # Action cost coefficient
        'e_power': 2,  # Diminishing returns to effort
        'e_lambda': 2,  # Exponential cost scaling
        'c_lambda': 50.0,  # Insurer fixed costs
        'm_gamma': 30.0,  # Monitoring technology costs
        'u_rho': 0.001,  # Risk aversion coefficient
    }
    
    print("="*60)
    print("DUOPOLY INSURANCE MODEL DEMONSTRATION")
    print("="*60)

    # Create logger for analysis
    logger = SimulationLogger(
        experiment_name="duopoly_insurance_model",
        log_level="INFO"
    )
    
    # Create function configuration
    # function_config = {
    #     'p': 'binomial',        # Use linear accident probability
    #     'm': 'linear',
    #     'e': 'power',  # Use exponential action cost
    #     'u': 'exponential',  # Use logarithmic utility function
    #     'f': 'binary_states', # Use binary state density
    #     'c': 'linear'
    # }

    function_config = {
        'p': 'binomial',        # Use binomial accident probability
        'm': 'linear',
        'e': 'power',  # Use power action cost
        'u': 'exponential',
        'f': 'binomial_states', # Use binomial state density
        'c': 'linear'
    }
    
    # Create function set
    functions = Functions(function_config)
    
    # Run KKT-based simulation
    print("\n" + "="*40)
    print("KKT-BASED SOLVER")
    print("="*40)
    
    solver = DuopolySolverKKT(functions, params)

    print("\n" + "="*40)
    
    # Try multistart optimization when regular solver fails
    multistart_success, multistart_solution = solver.multistart_solve(
        solver_name='ipopt',  # Use IPOPT as the underlying solver
        n_starts=2000,  # Number of multistart iterations
        verbose=True,
        save_plots=True,
        logger=logger,
        executable_path=None,  # Let Pyomo find the solver automatically
        seed=42,
        save_model=True,  # Save the model in .nl format
        model_filename='duopoly_insurance_model'  # Optional: specify filename
    )
    
    if multistart_success:
        print("✅ Multistart optimization successful!")
        print(f"Number of equilibrium solutions found: {len(multistart_solution)}")
    else:
        print("❌ Multistart optimization failed!")
        print("Consider adjusting model parameters.")


if __name__ == "__main__":
    main()
