"""
Solver Module

Contains the DuopolySolver class for solving the duopoly insurance model
using KKT conditions and Pyomo optimization.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple
import pyomo.environ as pyo

# Pyomo imports for KKT-based solving
try:
    from pyomo.opt import SolverStatus, TerminationCondition
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    print("Warning: Pyomo not available. KKT-based solving will not work.")

from solver.kkt import build_and_solve_model, run


class DuopolySolver:
    """
    Numerical solver for the duopoly insurance model using KKT conditions.
    """
    
    def __init__(self, functions, params: Dict):
        self.functions = functions
        self.params = params
        
        # Extract key parameters with error checking
        if 'W' not in params:
            raise ValueError("Parameter 'W' (initial wealth) is required")
        if 's' not in params:
            raise ValueError("Parameter 's' (accident severity) is required")
        if 'N' not in params:
            raise ValueError("Parameter 'N' (number of customers) is required")
        if 'delta1' not in params:
            raise ValueError("Parameter 'delta1' (insurer 1 monitoring level) is required")
        if 'delta2' not in params:
            raise ValueError("Parameter 'delta2' (insurer 2 monitoring level) is required")
        if 'theta_min' not in params:
            raise ValueError("Parameter 'theta_min' (minimum risk type) is required")
        if 'theta_max' not in params:
            raise ValueError("Parameter 'theta_max' (maximum risk type) is required")
        if 'n_theta' not in params:
            raise ValueError("Parameter 'n_theta' (number of risk types) is required")
        
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
        
        # Get state space for reference (will be computed dynamically)
        self.state_space = self.functions.f(1.0, 1.0, self.params)
        self.z_values, self.z_probs = self.state_space.get_all_states()
        self.n_states = len(self.z_values)
    
    def compute_reservation_utility(self, theta: float) -> float:
        """Compute reservation utility V_0(theta)."""
        def objective(a):
            p_no_accident = self.functions.p(a, self.params)
            p_accident = 1 - p_no_accident
            e_val = self.functions.e(a, theta, self.params)
            
            utility_no_accident = self.functions.u(self.W, self.params)
            utility_accident = self.functions.u(self.W - self.s, self.params)
            
            expected_utility = p_no_accident * utility_no_accident + p_accident * utility_accident - e_val
            return -expected_utility
        
        result = minimize(objective, x0=(self.a_min + self.a_max)/2, bounds=[(self.a_min, self.a_max)])
        return 15#-result.fun  # Return the actual optimal utility value
    
    def compute_expected_utility(self, a: float, phi1: float, phi2_values: np.ndarray, delta: float, theta: float) -> float:
        """
        Compute expected utility for given contract using discrete indemnity values.
        
        Args:
            a: Action level
            phi1: Premium
            phi2_values: Array of indemnity values for each state (length n_states)
            delta: Monitoring level
            theta: Risk type
        """
        p_no_accident = self.functions.p(a, self.params)  # Probability of no accident
        p_accident = 1 - p_no_accident  # Probability of accident
        e_val = self.functions.e(a, theta, self.params)

        # Get discrete state space for this action and monitoring level
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        
        # Expected utility when accident occurs (sum over discrete states)
        expected_utility_accident = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_val = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            expected_utility_accident += u_val * z_probs[i]
        
        # Expected utility when no accident occurs
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        m_val = self.functions.m(delta, self.params)
        
        return p_no_accident * utility_no_accident + p_accident * expected_utility_accident - e_val - m_val
    
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
    
    def compute_choice_probabilities(self, theta: float, 
                                   a1: float, phi1_1: float, phi2_1: np.ndarray,
                                   a2: float, phi1_2: float, phi2_2: np.ndarray) -> Tuple[float, float, float]:
        """Compute choice probabilities P_0(θ), P_1(θ), P_2(θ) using multinomial logit."""
        V0 = self.compute_reservation_utility(theta)
        V1 = self.compute_utility_Vi(theta, a1, phi1_1, phi2_1, self.delta1)
        V2 = self.compute_utility_Vi(theta, a2, phi1_2, phi2_2, self.delta2)
        
        utilities = [V0, V1, V2]
        scaled_utilities = [u / self.mu for u in utilities]
        
        # Compute choice probabilities using Pyomo exp
        exp_utilities = [pyo.exp(u) for u in scaled_utilities]
        denom = sum(exp_utilities)
        
        P0 = exp_utilities[0] / denom
        P1 = exp_utilities[1] / denom
        P2 = exp_utilities[2] / denom
        
        return P0, P1, P2
    
    def compute_dPi_da(self, theta: float, i: int,
                       a1: float, phi1_1: float, phi2_1: np.ndarray,
                       a2: float, phi1_2: float, phi2_2: np.ndarray) -> float:
        """Compute derivative of choice probability P_i with respect to action a_i."""
        P0, P1, P2 = self.compute_choice_probabilities(theta, a1, phi1_1, phi2_1, a2, phi1_2, phi2_2)
        
        if i == 1:
            dVi_da = self.compute_G_function(theta, a1, phi1_1, phi2_1, self.delta1)
            Pi = P1
        else:
            dVi_da = self.compute_G_function(theta, a2, phi1_2, phi2_2, self.delta2)
            Pi = P2
        
        return (1 / self.mu) * Pi * (1 - Pi) * dVi_da
    
    def compute_dVi_da(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute derivative of V_i with respect to action a."""
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        dp_da = self.functions.dp_da(a, self.params)
        de_da = self.functions.de_da(a, theta, self.params)
        
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        
        integral1 = 0.0
        integral2 = 0.0
        
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            integral1 += u_accident * z_probs[i]
            integral2 += u_accident * df_da[i]
        
        return dp_da * (utility_no_accident - integral1) + p_accident * integral2 - de_da
    
    def compute_dVi_dphi1(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute derivative of V_i with respect to premium φ₁."""
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        
        du_dphi1_no_accident = -self.functions.du_dx(self.W - phi1, self.params)
        
        integral_du_dphi1_accident = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            du_dphi1_accident = -self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
            integral_du_dphi1_accident += du_dphi1_accident * z_probs[i]
        
        return p_no_accident * du_dphi1_no_accident + p_accident * integral_du_dphi1_accident
    
    def compute_dVi_dphi2(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float, z_idx: int) -> float:
        """Compute derivative of V_i with respect to indemnity φ₂(z)."""
        p_accident = 1 - self.functions.p(a, self.params)
        
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        
        phi2_val = phi2_values[z_idx]
        du_dphi2 = self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
        f_z = z_probs[z_idx]
        
        return p_accident * du_dphi2 * f_z
    
    def compute_dPi_dphi1(self, theta: float, i: int,
                          a1: float, phi1_1: float, phi2_1: np.ndarray,
                          a2: float, phi1_2: float, phi2_2: np.ndarray) -> float:
        """Compute derivative of choice probability P_i with respect to premium φ₁^i."""
        P0, P1, P2 = self.compute_choice_probabilities(theta, a1, phi1_1, phi2_1, a2, phi1_2, phi2_2)
        
        if i == 1:
            dVi_dphi1 = self.compute_dVi_dphi1(theta, a1, phi1_1, phi2_1, self.delta1)
            Pi = P1
        else:
            dVi_dphi1 = self.compute_dVi_dphi1(theta, a2, phi1_2, phi2_2, self.delta2)
            Pi = P2
        
        return (1 / self.mu) * Pi * (1 - Pi) * dVi_dphi1
    
    def compute_dPi_dphi2(self, theta: float, i: int, z_idx: int,
                          a1: float, phi1_1: float, phi2_1: np.ndarray,
                          a2: float, phi1_2: float, phi2_2: np.ndarray) -> float:
        """Compute derivative of choice probability P_i with respect to indemnity φ₂^i(z)."""
        P0, P1, P2 = self.compute_choice_probabilities(theta, a1, phi1_1, phi2_1, a2, phi1_2, phi2_2)
        
        if i == 1:
            dVi_dphi2 = self.compute_dVi_dphi2(theta, a1, phi1_1, phi2_1, self.delta1, z_idx)
            Pi = P1
        else:
            dVi_dphi2 = self.compute_dVi_dphi2(theta, a2, phi1_2, phi2_2, self.delta2, z_idx)
            Pi = P2
        
        return (1 / self.mu) * Pi * (1 - Pi) * dVi_dphi2
    
    def compute_G_function(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute the G(θ) function from the mathematical model."""
        dp_da = self.functions.dp_da(a, self.params)
        de_da = self.functions.de_da(a, theta, self.params)
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        
        integral_1 = 0.0
        integral_2 = 0.0
        
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            integral_1 += u_accident * z_probs[i]
            integral_2 += u_accident * df_da[i]
        
        return dp_da * (utility_no_accident - integral_1) + p_accident * integral_2 - de_da
    
    def compute_dG_dphi1(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute derivative of G(θ) with respect to φ₁."""
        dp_da = self.functions.dp_da(a, self.params)
        p_accident = 1 - self.functions.p(a, self.params)
        
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        du_no_accident = self.functions.du_dx(self.W - phi1, self.params)
        
        integral_marginal_1 = 0.0
        integral_marginal_2 = 0.0
        
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            du_accident = self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
            integral_marginal_1 += du_accident * z_probs[i]
            integral_marginal_2 += du_accident * df_da[i]
        
        return -dp_da * (du_no_accident - integral_marginal_1) - p_accident * integral_marginal_2
    
    def compute_dG_dphi2(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float, z_idx: int) -> float:
        """Compute derivative of G(θ) with respect to φ₂(z)."""
        dp_da = self.functions.dp_da(a, self.params)
        p_accident = 1 - self.functions.p(a, self.params)
        
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        phi2_val = phi2_values[z_idx]
        du_accident = self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
        f_z = z_probs[z_idx]
        df_da_z = df_da[z_idx]
        
        return - dp_da * (du_accident * f_z) + p_accident * du_accident * df_da_z
    
    def compute_dG_da(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute derivative of G(θ) with respect to action a (second-order derivative)."""
        dp_da = self.functions.dp_da(a, self.params)
        d2p_da2 = self.functions.d2p_da2(a, self.params)
        d2e_da2 = self.functions.d2e_da2(a, theta, self.params)
        
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        # For discrete binary states, second derivatives are mathematically zero
        d2f_da2 = np.zeros_like(df_da)
        
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        
        integral_u_f = 0.0
        integral_u_df_da = 0.0
        integral_u_d2f_da2 = 0.0
        
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            
            integral_u_f += u_accident * z_probs[i]
            integral_u_df_da += u_accident * df_da[i]
            integral_u_d2f_da2 += u_accident * d2f_da2[i]
        
        term1 = d2p_da2 * (utility_no_accident - integral_u_f)
        term2 = -2 * dp_da * integral_u_df_da
        term3 = p_accident * integral_u_d2f_da2
        term4 = -d2e_da2
        
        return term1 + term2 + term3 + term4
    
    # Add the large methods as class methods
    def build_and_solve_model(self, solver_name='ipopt', verbose=False, debug_mode=True, executable_path=None):
        """Solve the duopoly equilibrium using KKT conditions with pyomo."""
        return build_and_solve_model(self, solver_name, verbose, debug_mode, executable_path)
    
    def run(self, solver_name='ipopt', verbose=True, save_plots=True, logger=None, executable_path=None):
        """Run KKT-based simulation for duopoly insurance model."""
        return run(self, solver_name, verbose, save_plots, logger, executable_path) 