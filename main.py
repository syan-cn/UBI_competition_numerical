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
    
    # Example parameters
    params = {
        'W': 1000.0,            # Initial wealth
        's': 600.0,             # Accident severity
        'N': 100,              # Number of customers
        'delta1': 0.7,         # Insurer 1 monitoring level
        'delta2': 0.2,         # Insurer 2 monitoring level
        'theta_min': 1,      # Minimum risk type
        'theta_max': 10,      # Maximum risk type
        'n_theta': 10,          # Number of risk types
        'mu': 500.0,             # Logit model scale parameter
        # 'p_alpha': 0.0,        # No accident probability parameter (for linear)
        # 'p_beta': 1.0,         # No accident probability parameter (for linear)
        'p_hat': 0.05,          # Base probability parameter (for binomial)
        'n_trials': 10,         # Number of trials (for binomial)
        'e_kappa': 100.0,       # Action cost parameter
        'e_power': 2.0,        # Action cost power
        # 'f_p_base': 0.5,       # State density parameter (for binary_states)
        'c_lambda': 100.0,       # Insurer cost parameter
        'm_gamma': 100.0,        # Monitoring cost parameter
        'u_rho': 1e-3,          # Utility parameter
        # 'u_max_val': 5000.0,        # Utility parameter
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
    #     'p': 'linear',        # Use linear accident probability
    #     'm': 'linear',
    #     'e': 'power',
    #     'u': 'exponential',
    #     'f': 'binary_states', # Use binary state density
    #     'c': 'linear'
    # }

    function_config = {
        'p': 'binomial',        # Use binomial accident probability
        'm': 'linear',
        'e': 'power',
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
        n_starts=2,  # Number of multistart iterations
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
