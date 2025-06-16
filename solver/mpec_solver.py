# MPEC Solver

"""
MPEC Solver Module

Contains the MPECSolver class for solving the duopoly insurance model
using Mathematical Programs with Equilibrium Constraints (MPEC).
"""

import numpy as np
import pyomo.environ as pyo
from pathlib import Path
import time
from typing import Dict, Tuple, Optional
from utils.config import get_solver_options
from utils.plotting import plot_results

# Pyomo imports for MPEC solving
try:
    from pyomo.opt import SolverStatus, TerminationCondition
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    print("Warning: Pyomo not available. MPEC solving will not work.")


class MPECSolver:
    """
    MPEC solver for the duopoly insurance model.
    
    This solver implements the mathematical formulation as an MPEC where:
    - Each insurer maximizes profit subject to incentive compatibility constraints
    - The equilibrium is characterized by the first-order conditions
    - Choice probabilities follow multinomial logit structure
    """
    
    def __init__(self, functions, params: Dict):
        self.functions = functions
        self.params = params
        
        # Extract key parameters with error checking
        required_params = ['W', 's', 'N', 'delta1', 'delta2', 'theta_min', 'theta_max', 'n_theta', 'mu']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Parameter '{param}' is required")
        
        self.W = params['W']
        self.s = params['s']
        self.N = params['N']
        self.delta1 = params['delta1']
        self.delta2 = params['delta2']
        self.mu = params['mu']
        
        self.theta_min = params['theta_min']
        self.theta_max = params['theta_max']
        self.n_theta = params['n_theta']
        
        # Action bounds - default to [0, 1] if not specified
        self.a_min = params.get('a_min', 0.0)
        self.a_max = params.get('a_max', 1.0)
        
        # Create risk type grid
        self.theta_grid = np.linspace(self.theta_min, self.theta_max, self.n_theta)
        
        # Risk type distribution (uniform by default)
        self.h_theta = np.ones(self.n_theta) / self.n_theta
        
        # Get state space for reference
        self.state_space = self.functions.f(1.0, 1.0, self.params)
        self.z_values, self.z_probs = self.state_space.get_all_states()
        self.n_states = len(self.z_values)
    
    def compute_reservation_utility(self, theta: float) -> float:
        """Compute reservation utility V_0(theta) = max_a { p(a)u(W) + (1-p(a))u(W-s) - e(a,θ) }."""
        from scipy.optimize import minimize
        
        def objective(a):
            p_no_accident = self.functions.p(a, self.params)
            p_accident = 1 - p_no_accident
            e_val = self.functions.e(a, theta, self.params)
            
            utility_no_accident = self.functions.u(self.W, self.params)
            utility_accident = self.functions.u(self.W - self.s, self.params)
            
            expected_utility = p_no_accident * utility_no_accident + p_accident * utility_accident - e_val
            return -expected_utility
        
        result = minimize(objective, x0=(self.a_min + self.a_max)/2, bounds=[(self.a_min, self.a_max)])
        return -result.fun
    
    def compute_utility_Vi(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """
        Compute utility V_i(θ) for a given consumer type θ and contract (a, φ1, φ2).
        
        Mathematical formulation:
        V_k(θ) = p(a^k(θ)) u(W-φ₁^k) + [1-p(a^k(θ))] ∫ u(W-φ₁^k+φ₂^k(z)-s) f(z|a^k(θ),δ^k) dz - e(a^k(θ),θ) - m(δ^k)
        """
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        e_val = self.functions.e(a, theta, self.params)
        m_val = self.functions.m(delta, self.params)
        
        # Get state space
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        
        # Utility when no accident occurs
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        
        # Expected utility when accident occurs
        expected_utility_accident = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            expected_utility_accident += u_accident * z_probs[i]
        
        return p_no_accident * utility_no_accident + p_accident * expected_utility_accident - e_val - m_val
    
    def compute_expected_utility(self, a: float, phi1: float, phi2_values: np.ndarray, delta: float, theta: float) -> float:
        """
        Alias for compute_utility_Vi to maintain compatibility with plotting function.
        """
        return self.compute_utility_Vi(theta, a, phi1, phi2_values, delta)
    
    def compute_choice_probabilities(self, theta: float, 
                                   a1: float, phi1_1: float, phi2_1: np.ndarray,
                                   a2: float, phi1_2: float, phi2_2: np.ndarray) -> Tuple[float, float, float]:
        """Compute choice probabilities P_0(θ), P_1(θ), P_2(θ) using multinomial logit."""
        V0 = self.compute_reservation_utility(theta)
        V1 = self.compute_utility_Vi(theta, a1, phi1_1, phi2_1, self.delta1)
        V2 = self.compute_utility_Vi(theta, a2, phi1_2, phi2_2, self.delta2)
        
        # Compute choice probabilities using multinomial logit
        exp_V0 = pyo.exp(V0 / self.mu)
        exp_V1 = pyo.exp(V1 / self.mu)
        exp_V2 = pyo.exp(V2 / self.mu)
        
        denominator = exp_V0 + exp_V1 + exp_V2
        
        P0 = exp_V0 / denominator
        P1 = exp_V1 / denominator
        P2 = exp_V2 / denominator
        
        return P0, P1, P2
    
    def compute_incentive_constraint(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """
        Compute the incentive compatibility constraint G(θ) = 0.
        
        G(θ) = ∂p(a^i(θ))/∂a^i(θ) [u(W-φ₁^i) - ∫ u(W-φ₁^i+φ₂^i(z)-s) f(z|a^i(θ),δ^i) dz]
               + (1-p(a^i(θ))) ∫ u(W-φ₁^i+φ₂^i(z)-s) ∂f(z|a^i(θ),δ^i)/∂a^i(θ) dz
               - ∂e(a^i(θ),θ)/∂a^i(θ)
        """
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        dp_da = self.functions.dp_da(a, self.params)
        de_da = self.functions.de_da(a, theta, self.params)
        
        # Get state space
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        # First term: ∂p(a^i(θ))/∂a^i(θ) [u(W-φ₁^i) - ∫ u(W-φ₁^i+φ₂^i(z)-s) f(z|a^i(θ),δ^i) dz]
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        
        integral_utility_accident = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            integral_utility_accident += u_accident * z_probs[i]
        
        first_term = dp_da * (utility_no_accident - integral_utility_accident)
        
        # Second term: (1-p(a^i(θ))) ∫ u(W-φ₁^i+φ₂^i(z)-s) ∂f(z|a^i(θ),δ^i)/∂a^i(θ) dz
        second_term = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            second_term += u_accident * df_da[i]
        second_term = p_accident * second_term
        
        return first_term + second_term - de_da
    
    def build_mpec_model(self) -> pyo.ConcreteModel:
        """Build the MPEC model for the duopoly equilibrium."""
        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo is required for MPEC solving. Please install it with: pip install pyomo")
        
        print("Building MPEC model for duopoly equilibrium...")
        
        # Create the optimization model
        model = pyo.ConcreteModel()
        
        # Sets
        model.THETA = pyo.Set(initialize=range(self.n_theta))  # Risk types
        model.Z = pyo.Set(initialize=range(self.n_states))     # States
        model.I = pyo.Set(initialize=[1, 2])                  # Insurers
        
        # Parameters
        model.W = pyo.Param(initialize=self.W)
        model.s = pyo.Param(initialize=self.s)
        model.N = pyo.Param(initialize=self.N)
        model.mu = pyo.Param(initialize=self.mu)
        model.theta_vals = pyo.Param(model.THETA, initialize={i: self.theta_grid[i] for i in range(self.n_theta)})
        model.h_theta = pyo.Param(model.THETA, initialize={i: self.h_theta[i] for i in range(self.n_theta)})
        model.z_vals = pyo.Param(model.Z, initialize={i: self.z_values[i] for i in range(self.n_states)})
        model.delta1 = pyo.Param(initialize=self.delta1)
        model.delta2 = pyo.Param(initialize=self.delta2)
        
        # Decision Variables for both insurers
        model.a = pyo.Var(model.I, model.THETA, domain=pyo.NonNegativeReals, 
                         bounds=(self.a_min, self.a_max))
        model.phi1 = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, self.s))
        model.phi2 = pyo.Var(model.I, model.Z, domain=pyo.NonNegativeReals, bounds=(0, self.s))
        
        # Objective function: maximize total profit
        def objective_rule(model):
            total_profit = 0.0
            
            for i in model.I:
                delta_val = model.delta1 if i == 1 else model.delta2
                c_delta = self.functions.c(delta_val, self.params)
                
                for t in model.THETA:
                    theta = model.theta_vals[t]
                    h_theta_val = model.h_theta[t]
                    a_val = model.a[i, t]
                    phi1_val = model.phi1[i]
                    
                    # Get phi2 values for all states
                    phi2_values = np.array([model.phi2[i, z] for z in model.Z])
                    
                    # Get contract values for the other insurer
                    other_insurer = 2 if i == 1 else 1
                    a_other = model.a[other_insurer, t]
                    phi1_other = model.phi1[other_insurer]
                    phi2_other_values = np.array([model.phi2[other_insurer, z] for z in model.Z])
                    
                    # Compute choice probability
                    P0, P1, P2 = self.compute_choice_probabilities(theta, a_val, phi1_val, phi2_values, 
                                                                  a_other, phi1_other, phi2_other_values)
                    Pi = P1 if i == 1 else P2
                    
                    # Compute profit term: φ₁^i - (1-p(a^i(θ))) ∫ φ₂^i(z) f(z|a^i(θ),δ^i) dz
                    p_no_accident = self.functions.p(a_val, self.params)
                    p_accident = 1 - p_no_accident
                    
                    # Get state space and compute integral
                    state_space = self.functions.f(a_val, delta_val, self.params)
                    z_values, z_probs = state_space.get_all_states()
                    
                    integral_phi2_f = 0.0
                    for j, z in enumerate(z_values):
                        integral_phi2_f += phi2_values[j] * z_probs[j]
                    
                    profit_term = phi1_val - p_accident * integral_phi2_f
                    
                    total_profit += self.N * Pi * profit_term * h_theta_val
                
                total_profit -= c_delta
            
            return total_profit
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
        
        # Incentive compatibility constraints
        def incentive_constraint_rule(model, i, t):
            """Incentive compatibility constraint G(θ) = 0."""
            theta = model.theta_vals[t]
            a_val = model.a[i, t]
            phi1_val = model.phi1[i]
            delta_val = model.delta1 if i == 1 else model.delta2
            
            # Get phi2 values for all states
            phi2_values = np.array([model.phi2[i, z] for z in model.Z])
            
            # Compute incentive constraint
            G_val = self.compute_incentive_constraint(theta, a_val, phi1_val, phi2_values, delta_val)
            
            return G_val == 0
        
        model.incentive_constraint = pyo.Constraint(model.I, model.THETA, rule=incentive_constraint_rule)
        
        return model
    
    def solve_mpec(self, solver_name='knitroampl', verbose=False, debug_mode=False, 
                   executable_path=None) -> Tuple[bool, Dict]:
        """Solve the MPEC model."""
        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo is required for MPEC solving")
        
        print(f"Solving MPEC model with {solver_name}...")
        start_time = time.time()
        
        # Build the model
        model = self.build_mpec_model()
        
        # Get solver options
        solver_options = get_solver_options(solver_name, verbose, debug_mode)
        
        # Create solver
        if executable_path:
            solver = pyo.SolverFactory(solver_name, executable=executable_path)
        else:
            solver = pyo.SolverFactory(solver_name)
        
        # Set solver options
        for key, value in solver_options.items():
            if value is not None:
                solver.options[key] = value
        
        # Solve the model
        try:
            results = solver.solve(model, tee=verbose)
            solve_time = time.time() - start_time
            
            # Check solution status
            if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
                print(f"✅ MPEC solved successfully in {solve_time:.2f} seconds")
                
                # Extract solution
                solution = self.extract_solution(model)
                solution['solve_time'] = solve_time
                solution['solver_status'] = str(results.solver.status)
                solution['termination_condition'] = str(results.solver.termination_condition)
                solution['objective_value'] = pyo.value(model.objective)
                
                return True, solution
            else:
                print(f"❌ MPEC solver failed: {results.solver.termination_condition}")
                return False, {'error': str(results.solver.termination_condition)}
                
        except Exception as e:
            print(f"❌ Error solving MPEC: {str(e)}")
            return False, {'error': str(e)}
    
    def extract_solution(self, model: pyo.ConcreteModel) -> Dict:
        """Extract solution from the solved model."""
        solution = {
            'a': {},
            'phi1': {},
            'phi2': {},
            'choice_probabilities': {},
            'utilities': {},
            'profits': {}
        }
        
        # Extract decision variables
        for i in model.I:
            solution['phi1'][i] = pyo.value(model.phi1[i])
            solution['phi2'][i] = {}
            for z in model.Z:
                solution['phi2'][i][z] = pyo.value(model.phi2[i, z])
            
            solution['a'][i] = {}
            for t in model.THETA:
                solution['a'][i][t] = pyo.value(model.a[i, t])
        
        # Compute additional quantities
        for t in model.THETA:
            theta = self.theta_grid[t]
            
            # Get values for both insurers
            a1 = solution['a'][1][t]
            phi1_1 = solution['phi1'][1]
            phi2_1 = np.array([solution['phi2'][1][z] for z in model.Z])
            
            a2 = solution['a'][2][t]
            phi1_2 = solution['phi1'][2]
            phi2_2 = np.array([solution['phi2'][2][z] for z in model.Z])
            
            # Compute choice probabilities
            P0, P1, P2 = self.compute_choice_probabilities(theta, a1, phi1_1, phi2_1, a2, phi1_2, phi2_2)
            solution['choice_probabilities'][t] = {'P0': P0, 'P1': P1, 'P2': P2}
            
            # Compute utilities
            V0 = self.compute_reservation_utility(theta)
            V1 = self.compute_utility_Vi(theta, a1, phi1_1, phi2_1, self.delta1)
            V2 = self.compute_utility_Vi(theta, a2, phi1_2, phi2_2, self.delta2)
            solution['utilities'][t] = {'V0': V0, 'V1': V1, 'V2': V2}
        
        # Convert to format expected by plotting function
        plotting_solution = {
            'insurer1': {
                'phi1': solution['phi1'][1],
                'phi2': np.array([solution['phi2'][1][z] for z in model.Z]),
                'a_schedule': np.array([solution['a'][1][t] for t in model.THETA])
            },
            'insurer2': {
                'phi1': solution['phi1'][2],
                'phi2': np.array([solution['phi2'][2][z] for z in model.Z]),
                'a_schedule': np.array([solution['a'][2][t] for t in model.THETA])
            }
        }
        
        # Add the original solution data for backward compatibility
        plotting_solution.update(solution)
        
        return plotting_solution
    
    def run(self, solver_name='knitroampl', verbose=True, save_plots=True, 
            logger=None, executable_path=None) -> Tuple[bool, Dict]:
        """Main method to run the MPEC solver."""
        print("="*60)
        print("MPEC SOLVER FOR DUOPOLY INSURANCE MODEL")
        print("="*60)
        
        # Solve the MPEC
        success, solution = self.solve_mpec(solver_name, verbose, False, executable_path)
        
        if success and save_plots:
            try:
                # Create plots directory
                plots_dir = Path("plots")
                plots_dir.mkdir(exist_ok=True)
                
                # Save plot to file
                plot_path = plots_dir / "mpec_results.png"
                plot_results(self, solution, save_path=str(plot_path))
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")
        
        return success, solution


def build_and_solve_mpec(functions, params: Dict, solver_name='knitroampl', 
                        verbose=False, debug_mode=False, executable_path=None) -> Tuple[bool, Dict]:
    """Convenience function to build and solve MPEC model."""
    solver = MPECSolver(functions, params)
    return solver.solve_mpec(solver_name, verbose, debug_mode, executable_path)


def run_mpec(functions, params: Dict, solver_name='knitroampl', verbose=True, 
             save_plots=True, logger=None, executable_path=None) -> Tuple[bool, Dict]:
    """Convenience function to run MPEC solver."""
    solver = MPECSolver(functions, params)
    return solver.run(solver_name, verbose, save_plots, logger, executable_path)
