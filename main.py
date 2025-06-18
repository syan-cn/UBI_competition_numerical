"""
Main Module for Duopoly Insurance Model

This module provides the main entry point and example usage for the modularized
duopoly insurance model framework.
"""

from elements.function_set import Functions
from solver.helper import DuopolySolver
from utils.logger import SimulationLogger


def main():
    """
    Main function demonstrating the duopoly insurance model.
    
    """
    
    # Example parameters
    params = {
        'W': 500.0,            # Initial wealth
        's': 200.0,             # Accident severity
        'N': 100,              # Number of customers
        'delta1': 1,         # Insurer 1 monitoring level
        'delta2': 0.3,         # Insurer 2 monitoring level
        'theta_min': 0.05,      # Minimum risk type
        'theta_max': 1,      # Maximum risk type
        'n_theta': 10,          # Number of risk types
        'a_min': 0.05,          # Minimum action level
        'a_max': 0.95,          # Maximum action level
        'mu': 50.0,             # Logit model scale parameter
        'p_alpha': 0.0,        # No accident probability parameter (for linear)
        'p_beta': 1.0,         # No accident probability parameter (for linear)
        'p_hat': 0.4,          # Base probability parameter (for binomial)
        'n_trials': 2,         # Number of trials (for binomial)
        'e_kappa': 15.0,       # Action cost parameter
        'e_power': 2.0,        # Action cost power
        'f_p_base': 0.5,       # State density parameter (for binary_states)
        'c_lambda': 10.0,       # Insurer cost parameter
        'm_gamma': 5.0,        # Monitoring cost parameter
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
    
    solver = DuopolySolver(functions, params)

    print("\n" + "="*40)
    
    # Try multistart optimization when regular solver fails
    multistart_success, multistart_solution = solver.multistart_solve(
        solver_name='knitroampl',
        n_starts=10,
        verbose=True,
        save_plots=True,
        logger=logger,
        executable_path='/Users/syan/knitro-14.2.0-ARM-MacOS/knitroampl/knitroampl',
        seed=42
    )
    
    if multistart_success:
        print("✅ Multistart optimization successful!")
        print(f"Number of equilibrium solutions found: {len(multistart_solution)}")

        # Display summary of first solution as representative
        if multistart_solution:
            first_solution = multistart_solution[0]['solution']
            print(f"\nRepresentative equilibrium solution:")
            print(f"  Insurer 1 premium: {first_solution['insurer1']['phi1']:.4f}")
            print(f"  Insurer 2 premium: {first_solution['insurer2']['phi1']:.4f}")
    else:
        print("❌ Multistart optimization failed!")
        print("Consider adjusting model parameters.")


if __name__ == "__main__":
    main()
