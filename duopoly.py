"""
Duopoly Insurance Model Framework

A flexible, modular framework for solving duopoly-style insurance models
with different functional forms and parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root_scalar
from typing import Dict, Tuple, List
from logger import SimulationLogger
from pathlib import Path
import datetime
from itertools import product
import multiprocessing as mp
from functools import partial
import time

# ============================================================================
# DISCRETE STATE SPACE
# ============================================================================

class DiscreteStateSpace:
    """Base class for discrete state spaces in insurance models."""
    
    def __init__(self, z_values: List[float], z_probs: List[float]):
        """
        Initialize discrete state space.
        
        Args:
            z_values: List of possible state values
            z_probs: List of probabilities for each state (must sum to 1)
        """
        if z_probs is None:
            raise ValueError("z_probs is required and cannot be None")
            
        self.z_values = np.array(z_values)
        self.n_states = len(z_values)
        
        self.z_probs = np.array(z_probs)
        # Normalize to ensure sum = 1
        self.z_probs = self.z_probs / np.sum(self.z_probs)
    
    def get_probability(self, z: float) -> float:
        """Get probability for a specific state value."""
        if z in self.z_values:
            idx = np.where(self.z_values == z)[0][0]
            return self.z_probs[idx]
        return 0.0
    
    def get_all_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all state values and their probabilities."""
        return self.z_values, self.z_probs


class StateDensity:
    """Discrete state density functions f(z | a, delta)."""
    
    @staticmethod
    def binary_states(a: float, delta: float, params: Dict) -> DiscreteStateSpace:
        """
        Binary state space: z ∈ {0, 1}
        P(z=1 | a, delta) = (1 - delta) * p_base + a * delta
        """
        p_base = params.get('f_p_base', 0.5)
        
        prob_1 = max(0.0, min(1.0, (1 - delta) * p_base + a * delta))
        prob_0 = 1.0 - prob_1
        
        return DiscreteStateSpace([0.0, 1.0], [prob_0, prob_1])

    @staticmethod
    def df_da_binary_states(a: float, delta: float, params: Dict) -> np.ndarray:
        # dP(z=1)/da = delta, dP(z=0)/da = -delta
        return np.array([-delta, delta])

    @staticmethod
    def custom_discrete(a: float, delta: float, params: Dict) -> DiscreteStateSpace:
        """
        Custom discrete distribution with user-defined states and probabilities
        """
        if 'f_z_values' not in params:
            raise ValueError("Parameter 'f_z_values' is required for custom_discrete state density function")
        if 'f_base_probs' not in params:
            raise ValueError("Parameter 'f_base_probs' is required for custom_discrete state density function")
        if 'f_p_a' not in params:
            raise ValueError("Parameter 'f_p_a' is required for custom_discrete state density function")
        if 'f_p_delta' not in params:
            raise ValueError("Parameter 'f_p_delta' is required for custom_discrete state density function")
        
        z_values = params['f_z_values']
        base_probs = params['f_base_probs']
        
        # Adjust probabilities based on action and monitoring
        p_a = params['f_p_a']
        p_delta = params['f_p_delta']
        
        adjusted_probs = []
        for i, base_prob in enumerate(base_probs):
            # Higher states get higher probability with more action/monitoring
            adjustment = (i / len(base_probs)) * (p_a * a + p_delta * delta)
            adjusted_prob = max(0.0, base_prob + adjustment)
            adjusted_probs.append(adjusted_prob)
        
        # Normalize
        adjusted_probs = np.array(adjusted_probs)
        adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
        
        return DiscreteStateSpace(z_values, adjusted_probs)

    @staticmethod
    def df_da_custom_discrete(a: float, delta: float, params: Dict) -> np.ndarray:
        z_values = params['f_z_values']
        base_probs = np.array(params['f_base_probs'])
        p_a = params['f_p_a']
        p_delta = params['f_p_delta']
        n = len(base_probs)
        # Unnormalized derivative
        d_adjusted = np.array([(i / n) * p_a for i in range(n)])
        # Unnormalized adjusted_probs (same as in custom_discrete)
        adjusted_probs = np.array([max(0.0, base_probs[i] + (i / n) * (p_a * a + p_delta * delta)) for i in range(n)])
        sum_adj = np.sum(adjusted_probs)
        # Normalized derivative: (d_adjusted * sum_adj - adjusted_probs * sum(d_adjusted)) / sum_adj^2
        sum_d_adj = np.sum(d_adjusted)
        d_norm = (d_adjusted * sum_adj - adjusted_probs * sum_d_adj) / (sum_adj ** 2)
        return d_norm


# ============================================================================
# FUNCTION TEMPLATES
# ============================================================================

class FunctionTemplates:
    """Abstract function templates for the insurance model."""
    
    @staticmethod
    def p(a: float, params: Dict) -> float:
        """
        No accident probability function.
        
        Args:
            a: Action level
            params: Dictionary containing function parameters
            
        Returns:
            Probability of no accident
        """
        raise NotImplementedError("Subclass must implement p(a, params)")
    
    @staticmethod
    def m(delta: float, params: Dict) -> float:
        """
        Monitoring cost function.
        
        Args:
            delta: Monitoring level
            params: Dictionary containing function parameters
            
        Returns:
            Monitoring cost
        """
        raise NotImplementedError("Subclass must implement m(delta, params)")
    
    @staticmethod
    def e(a: float, theta: float, params: Dict) -> float:
        """
        Action cost function.
        
        Args:
            a: Action level
            theta: Driver's risk type
            params: Dictionary containing function parameters
            
        Returns:
            Action cost
        """
        raise NotImplementedError("Subclass must implement e(a, theta, params)")
    
    @staticmethod
    def u(x: float, params: Dict) -> float:
        """
        Utility function.
        
        Args:
            x: Wealth level
            params: Dictionary containing function parameters
            
        Returns:
            Utility value
        """
        raise NotImplementedError("Subclass must implement u(x, params)")
    
    @staticmethod
    def f(a: float, delta: float, params: Dict) -> DiscreteStateSpace:
        """
        State density function - now returns DiscreteStateSpace.
        
        Args:
            a: Action level
            delta: Monitoring level
            params: Dictionary containing function parameters
            
        Returns:
            DiscreteStateSpace object
        """
        raise NotImplementedError("Subclass must implement f(a, delta, params)")
    
    @staticmethod
    def c(delta: float, params: Dict) -> float:
        """
        Insurer's cost function.
        
        Args:
            delta: Monitoring level
            params: Dictionary containing function parameters
            
        Returns:
            Insurer's cost
        """
        raise NotImplementedError("Subclass must implement c(delta, params)")


# ============================================================================
# INDIVIDUAL FUNCTION IMPLEMENTATIONS
# ============================================================================

class NoAccidentProbability:
    """Different functional forms for no accident probability p(a)."""
    
    @staticmethod
    def linear(a: float, params: Dict) -> float:
        """Linear: p(a) = alpha + beta * a (no accident probability increases with action)"""
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for linear no accident probability function")
        if 'p_beta' not in params:
            raise ValueError("Parameter 'p_beta' is required for linear no accident probability function")
        
        alpha = params['p_alpha']
        beta = params['p_beta']
        return max(0, min(1, alpha + beta * a))
    
    @staticmethod
    def dp_da_linear(a: float, params: Dict) -> float:
        beta = params['p_beta']
        return beta
    
    @staticmethod
    def exponential(a: float, params: Dict) -> float:
        """Exponential: p(a) = 1 - alpha * exp(-beta * a) (no accident probability increases with action)"""
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for exponential no accident probability function")
        if 'p_beta' not in params:
            raise ValueError("Parameter 'p_beta' is required for exponential no accident probability function")
        
        alpha = params['p_alpha']
        beta = params['p_beta']
        return max(0, min(1, 1 - alpha * np.exp(-beta * a)))
    
    @staticmethod
    def dp_da_exponential(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        return alpha * beta * np.exp(-beta * a)
    
    @staticmethod
    def power(a: float, params: Dict) -> float:
        """Power: p(a) = 1 - alpha * (1 + a)^(-beta) (no accident probability increases with action)"""
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for power no accident probability function")
        if 'p_beta' not in params:
            raise ValueError("Parameter 'p_beta' is required for power no accident probability function")
        
        alpha = params['p_alpha']
        beta = params['p_beta']
        return max(0, min(1, 1 - alpha * (1 + a)**(-beta)))
    
    @staticmethod
    def dp_da_power(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        return alpha * beta * (1 + a) ** (-(beta + 1))
    
    @staticmethod
    def logistic(a: float, params: Dict) -> float:
        """Logistic: p(a) = 1 / (1 + exp(-beta * (a - alpha))) (no accident probability increases with action)"""
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for logistic no accident probability function")
        if 'p_beta' not in params:
            raise ValueError("Parameter 'p_beta' is required for logistic no accident probability function")
        
        alpha = params['p_alpha']
        beta = params['p_beta']
        return 1 / (1 + np.exp(-beta * (a - alpha)))
    
    @staticmethod
    def dp_da_logistic(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        exp_term = np.exp(-beta * (a - alpha))
        denom = (1 + exp_term) ** 2
        return beta * exp_term / denom


class MonitoringCost:
    """Different functional forms for monitoring cost m(delta)."""
    
    @staticmethod
    def linear(delta: float, params: Dict) -> float:
        """Linear: m(delta) = gamma * delta"""
        if 'm_gamma' not in params:
            raise ValueError("Parameter 'm_gamma' is required for linear monitoring cost function")
        
        gamma = params['m_gamma']
        return gamma * delta
    
    @staticmethod
    def exponential(delta: float, params: Dict) -> float:
        """Exponential: m(delta) = gamma * exp(delta)"""
        if 'm_gamma' not in params:
            raise ValueError("Parameter 'm_gamma' is required for exponential monitoring cost function")
        
        gamma = params['m_gamma']
        return gamma * np.exp(delta)
    
    @staticmethod
    def power(delta: float, params: Dict) -> float:
        """Power: m(delta) = gamma * delta^power"""
        if 'm_gamma' not in params:
            raise ValueError("Parameter 'm_gamma' is required for power monitoring cost function")
        if 'm_power' not in params:
            raise ValueError("Parameter 'm_power' is required for power monitoring cost function")
        
        gamma = params['m_gamma']
        power = params['m_power']
        return gamma * delta**power


class ActionCost:
    """Different functional forms for action cost e(a, theta)."""
    
    @staticmethod
    def linear(a: float, theta: float, params: Dict) -> float:
        """Linear: e(a, theta) = kappa * a / theta"""
        if 'e_kappa' not in params:
            raise ValueError("Parameter 'e_kappa' is required for linear action cost function")
        
        kappa = params['e_kappa']
        return kappa * a / max(theta, 0.1)
    
    @staticmethod
    def de_da_linear(a: float, theta: float, params: Dict) -> float:
        kappa = params['e_kappa']
        return kappa / max(theta, 0.1)
    
    @staticmethod
    def exponential(a: float, theta: float, params: Dict) -> float:
        """Exponential: e(a, theta) = kappa * exp(a) / theta"""
        if 'e_kappa' not in params:
            raise ValueError("Parameter 'e_kappa' is required for exponential action cost function")
        
        kappa = params['e_kappa']
        return kappa * np.exp(a) / max(theta, 0.1)
    
    @staticmethod
    def de_da_exponential(a: float, theta: float, params: Dict) -> float:
        kappa = params['e_kappa']
        return kappa * np.exp(a) / max(theta, 0.1)
    
    @staticmethod
    def power(a: float, theta: float, params: Dict) -> float:
        """Power: e(a, theta) = kappa * a^power * theta"""
        if 'e_kappa' not in params:
            raise ValueError("Parameter 'e_kappa' is required for power action cost function")
        if 'e_power' not in params:
            raise ValueError("Parameter 'e_power' is required for power action cost function")
        
        kappa = params['e_kappa']
        power = params['e_power']
        return kappa * a**power * theta
    
    @staticmethod
    def de_da_power(a: float, theta: float, params: Dict) -> float:
        kappa = params['e_kappa']
        power = params['e_power']
        return kappa * power * a ** (power - 1) * theta


class Utility:
    """Different functional forms for utility u(x)."""
    
    @staticmethod
    def linear(x: float, params: Dict) -> float:
        """Linear: u(x) = x"""
        return x
    
    @staticmethod
    def exponential(x: float, params: Dict) -> float:
        """Exponential: u(x) = 1 - exp(-rho * x)"""
        if 'u_rho' not in params:
            raise ValueError("Parameter 'u_rho' is required for exponential utility function")
        
        rho = params['u_rho']
        return 1 - np.exp(-rho * x)
    
    @staticmethod
    def power(x: float, params: Dict) -> float:
        """Power: u(x) = x^rho for rho < 1"""
        if 'u_rho' not in params:
            raise ValueError("Parameter 'u_rho' is required for power utility function")
        
        rho = params['u_rho']
        return x**rho if x > 0 else 0
    
    @staticmethod
    def logarithmic(x: float, params: Dict) -> float:
        """Logarithmic: u(x) = log(1 + x)"""
        return np.log(1 + x) if x > -1 else -np.inf


class InsurerCost:
    """Different functional forms for insurer cost c(delta)."""
    
    @staticmethod
    def linear(delta: float, params: Dict) -> float:
        """Linear: c(delta) = lambda * delta"""
        if 'c_lambda' not in params:
            raise ValueError("Parameter 'c_lambda' is required for linear insurer cost function")
        
        lambda_val = params['c_lambda']
        return lambda_val * delta
    
    @staticmethod
    def exponential(delta: float, params: Dict) -> float:
        """Exponential: c(delta) = lambda * exp(delta)"""
        if 'c_lambda' not in params:
            raise ValueError("Parameter 'c_lambda' is required for exponential insurer cost function")
        
        lambda_val = params['c_lambda']
        return lambda_val * np.exp(delta)
    
    @staticmethod
    def power(delta: float, params: Dict) -> float:
        """Power: c(delta) = lambda * delta^power"""
        if 'c_lambda' not in params:
            raise ValueError("Parameter 'c_lambda' is required for power insurer cost function")
        if 'c_power' not in params:
            raise ValueError("Parameter 'c_power' is required for power insurer cost function")
        
        lambda_val = params['c_lambda']
        power = params['c_power']
        return lambda_val * delta**power


# ============================================================================
# FLEXIBLE FUNCTION CONFIGURATOR
# ============================================================================

class FlexibleFunctions(FunctionTemplates):
    """
    Flexible function configurator that allows mixing different functional forms
    for each function independently.
    """
    
    def __init__(self, function_config: Dict):
        """
        Initialize with function configuration.
        
        Args:
            function_config: Dictionary specifying which functional form to use for each function
                           Example: {
                               'p': 'linear',
                               'm': 'linear', 
                               'e': 'power',
                               'u': 'logarithmic',
                               'f': 'binary_states',
                               'c': 'linear'
                           }
        """
        self.config = function_config
        
        # Function mappings
        self.p_functions = {
            'linear': NoAccidentProbability.linear,
            'exponential': NoAccidentProbability.exponential,
            'power': NoAccidentProbability.power,
            'logistic': NoAccidentProbability.logistic
        }
        
        self.m_functions = {
            'linear': MonitoringCost.linear,
            'exponential': MonitoringCost.exponential,
            'power': MonitoringCost.power
        }
        
        self.e_functions = {
            'linear': ActionCost.linear,
            'exponential': ActionCost.exponential,
            'power': ActionCost.power
        }
        
        self.u_functions = {
            'linear': Utility.linear,
            'exponential': Utility.exponential,
            'power': Utility.power,
            'logarithmic': Utility.logarithmic
        }
        
        self.f_functions = {
            'binary_states': StateDensity.binary_states,
            'custom_discrete': StateDensity.custom_discrete,
        }
        
        self.c_functions = {
            'linear': InsurerCost.linear,
            'exponential': InsurerCost.exponential,
            'power': InsurerCost.power
        }
    
    @staticmethod
    def choice_probabilities(V0, V1, V2, mu):
        """
        Compute choice probabilities using logit model.
        
        Args:
            V0: Reservation utilities (no insurance)
            V1: Utilities from insurer 1
            V2: Utilities from insurer 2
            mu: Scale parameter for logit model
            
        Returns:
            Tuple of (P0, P1, P2) - choice probabilities for each option
        """
        V0, V1, V2 = np.array(V0)/mu, np.array(V1)/mu, np.array(V2)/mu
        expV = np.exp(np.vstack([V0, V1, V2]))
        denom = expV.sum(axis=0)
        return expV / denom  # rows: 0,1,2
    
    def p(self, a: float, params: Dict) -> float:
        """No accident probability function."""
        func_type = self.config.get('p', 'linear')
        return self.p_functions[func_type](a, params)
    
    def m(self, delta: float, params: Dict) -> float:
        """Monitoring cost function."""
        func_type = self.config.get('m', 'linear')
        return self.m_functions[func_type](delta, params)
    
    def e(self, a: float, theta: float, params: Dict) -> float:
        """Action cost function."""
        func_type = self.config.get('e', 'power')
        return self.e_functions[func_type](a, theta, params)
    
    def u(self, x: float, params: Dict) -> float:
        """Utility function."""
        func_type = self.config.get('u', 'linear')
        return self.u_functions[func_type](x, params)
    
    def f(self, a: float, delta: float, params: Dict) -> DiscreteStateSpace:
        """State density function - now returns DiscreteStateSpace."""
        func_type = self.config.get('f', 'binary_states')
        return self.f_functions[func_type](a, delta, params)
    
    def c(self, delta: float, params: Dict) -> float:
        """Insurer cost function."""
        func_type = self.config.get('c', 'linear')
        return self.c_functions[func_type](delta, params)

    def dp_da(self, a: float, params: Dict) -> float:
        func_type = self.config.get('p', 'linear')
        return getattr(NoAccidentProbability, f"dp_da_{func_type}")(a, params)

    def de_da(self, a: float, theta: float, params: Dict) -> float:
        func_type = self.config.get('e', 'power')
        return getattr(ActionCost, f"de_da_{func_type}")(a, theta, params)

    def df_da(self, a: float, delta: float, params: Dict) -> np.ndarray:
        func_type = self.config.get('f', 'binary_states')
        return getattr(StateDensity, f"df_da_{func_type}")(a, delta, params)


# ============================================================================
# NUMERICAL SOLVER
# ============================================================================

class DuopolySolver:
    """
    Numerical solver for the duopoly insurance model with discrete state spaces.
    
    Action bounds are configurable via a_min and a_max parameters (default: [0, 1]).
    This allows the model to be adapted to different action space specifications.
    """
    
    def __init__(self, functions: FlexibleFunctions, params: Dict):
        """
        Initialize the solver.
        
        Args:
            functions: FlexibleFunctions instance
            params: Dictionary containing all model parameters
        """
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
        
        self.W = params['W']  # Initial wealth
        self.s = params['s']   # Accident severity
        self.N = params['N']   # Number of customers
        self.delta1 = params['delta1']  # Insurer 1's monitoring level
        self.delta2 = params['delta2']  # Insurer 2's monitoring level
        
        # Risk type grid parameters
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
            p_no_accident = self.functions.p(a, self.params)  # Probability of no accident
            p_accident = 1 - p_no_accident  # Probability of accident
            e_val = self.functions.e(a, theta, self.params)
            
            # Expected utility without insurance
            # When no accident occurs (prob p(a)): utility is u(W)
            # When accident occurs (prob 1-p(a)): utility is u(W - s)
            utility_no_accident = self.functions.u(self.W, self.params)
            utility_accident = self.functions.u(self.W - self.s, self.params)
            
            expected_utility = p_no_accident * utility_no_accident + p_accident * utility_accident - e_val
            return -expected_utility
        
        # Use configurable action bounds
        result = minimize(objective, x0=(self.a_min + self.a_max)/2, bounds=[(self.a_min, self.a_max)])
        return -result.fun
    
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
        m_val = self.functions.m(delta, self.params)
        
        # Get discrete state space for this action and monitoring level
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        
        # Expected utility when accident occurs (sum over discrete states)
        expected_utility_accident = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]  # Direct indemnity value for this state
            u_val = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            expected_utility_accident += u_val * z_probs[i]
        
        # Expected utility when no accident occurs
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        
        return p_accident * expected_utility_accident + \
               p_no_accident * utility_no_accident - m_val - e_val
    
    def incentive_constraint(self, a: float, phi1: float, phi2_values: np.ndarray, delta: float, theta: float) -> float:
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        dp_da = self.functions.dp_da(a, self.params)
        de_da = self.functions.de_da(a, theta, self.params)
        z_values, z_probs = self.functions.f(a, delta, self.params).get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        # Compute integrals for accident case
        integral1 = 0.0
        integral2 = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_val = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            f_val = z_probs[i]
            integral1 += u_val * f_val
            integral2 += u_val * df_da[i]
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        return dp_da * (utility_no_accident - integral1) + p_accident * integral2 - de_da
    
    def solve_contract(self, insurer_id: int, competitor_contract: Dict = None) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Solve for optimal contract maximizing expected profit with choice probabilities.
        
        Args:
            insurer_id: 1 or 2 for the insurer
            competitor_contract: Dictionary with competitor's contract details
            
        Returns:
            Tuple of (action schedule, premium, indemnity values array)
        """
        if insurer_id not in [1, 2]:
            raise ValueError("insurer_id must be 1 or 2")
            
        if insurer_id == 1:
            delta = self.delta1
        else:
            delta = self.delta2
        
        def objective(x):
            # x contains: [premium, indemnity_values..., actions...]
            phi1 = x[0]
            phi2_values = x[1:1+self.n_states]
            a_schedule = x[1+self.n_states:1+self.n_states+self.n_theta]
            
            # Compute expected profit with choice probabilities
            total_profit = 0.0
            
            for i, theta in enumerate(self.theta_grid):
                # Compute utilities for this theta
                V_own = self.compute_expected_utility(a_schedule[i], phi1, phi2_values, delta, theta)
                V0 = self.compute_reservation_utility(theta)
                
                # Competitor utility (if available)
                if competitor_contract:
                    V_comp = competitor_contract.get('utilities', [V0] * self.n_theta)[i]
                else:
                    V_comp = V0  # Default to reservation utility
                
                # Use logit choice probabilities for proper probabilistic choice modeling
                # This allows for smooth optimization and realistic market shares
                mu = self.params.get('xi_scale')
                if mu is None:
                    raise ValueError("Parameter 'xi_scale' is required for choice probability computation")
                    
                V0_scaled, V_own_scaled, V_comp_scaled = V0/mu, V_own/mu, V_comp/mu
                expV = np.exp([V0_scaled, V_own_scaled, V_comp_scaled])
                denom = expV.sum()
                prob_choose_own = expV[1] / denom  # Probability of choosing own insurer
                
                # Expected indemnity payment
                p_action = self.functions.p(a_schedule[i], self.params)
                state_space = self.functions.f(a_schedule[i], delta, self.params)
                _, z_probs = state_space.get_all_states()
                expected_indemnity = np.sum(phi2_values * z_probs) * (1 - p_action)
                
                # Profit per customer of this type
                profit_per_customer = phi1 - expected_indemnity - self.functions.c(delta, self.params)
                
                # Weight by choice probability and type distribution
                total_profit += prob_choose_own * profit_per_customer * self.h_theta[i]
            
            return -total_profit * self.N  # Negative for minimization
        
        def constraints(x):
            phi1 = x[0]
            phi2_values = x[1:1+self.n_states]
            a_schedule = x[1+self.n_states:1+self.n_states+self.n_theta]
            
            # Incentive constraints for each theta
            violations = []
            for i, theta in enumerate(self.theta_grid):
                violation = self.incentive_constraint(a_schedule[i], phi1, phi2_values, delta, theta)
                violations.append(violation)
            
            return violations
        
        # Initial guess: [premium, indemnity_values..., actions...]
        # Use reasonable initial values based on model parameters
        initial_premium = self.s * 0.5  # Start with half the accident severity
        initial_indemnity = self.s * 0.3  # Start with 30% of accident severity
        initial_action = (self.a_min + self.a_max) / 2  # Start with middle of action range
        
        x0 = np.concatenate([[initial_premium], np.ones(self.n_states) * initial_indemnity, np.ones(self.n_theta) * initial_action])
        
        # Bounds: [premium, indemnity_values..., actions...]
        bounds = [(0.1, self.s)] + [(0.0, self.s)] * self.n_states + [(self.a_min, self.a_max)] * self.n_theta
        
        # Solve optimization problem
        result = minimize(
            objective,
            x0=x0,
            constraints={'type': 'eq', 'fun': constraints},
            bounds=bounds,
            method='SLSQP'
        )
        
        if result.success:
            phi1 = result.x[0]
            phi2_values = result.x[1:1+self.n_states]
            a_schedule = result.x[1+self.n_states:1+self.n_states+self.n_theta]
            return a_schedule, phi1, phi2_values
        else:
            # Fallback to reasonable values if optimization fails
            a_schedule = np.ones(self.n_theta) * initial_action
            phi1 = initial_premium
            phi2_values = np.ones(self.n_states) * initial_indemnity
            return a_schedule, phi1, phi2_values
    
    def compute_reservation_utility_grid(self) -> np.ndarray:
        """Compute reservation utilities for all theta types."""
        V0 = np.zeros(self.n_theta)
        for i, theta in enumerate(self.theta_grid):
            V0[i] = self.compute_reservation_utility(theta)
        return V0
    
    def compute_utility_grid(self, insurer_id: int, a_schedule: np.ndarray, 
                           phi1: float, phi2_values: np.ndarray) -> np.ndarray:
        """
        Compute utilities for all theta types for a given insurer.
        
        Args:
            insurer_id: 1 or 2 for the insurer
            a_schedule: Action schedule (required)
            phi1: Premium (required)
            phi2_values: Indemnity values (required)
        """
        if insurer_id not in [1, 2]:
            raise ValueError("insurer_id must be 1 or 2")
            
        if a_schedule is None:
            raise ValueError("a_schedule is required")
        if phi1 is None:
            raise ValueError("phi1 is required")
        if phi2_values is None:
            raise ValueError("phi2_values is required")
        
        if insurer_id == 1:
            delta = self.delta1
        else:
            delta = self.delta2
        
        V = np.zeros(self.n_theta)
        for i, theta in enumerate(self.theta_grid):
            V[i] = self.compute_expected_utility(a_schedule[i], phi1, phi2_values, delta, theta)
        return V
    
    def compute_expected_profit(self, insurer_id: int, a_schedule: np.ndarray, 
                              phi1: float, phi2_values: np.ndarray, choice_probs: np.ndarray) -> float:
        """
        Compute expected profit for an insurer considering choice probabilities.
        
        Args:
            insurer_id: 1 or 2 for the insurer
            a_schedule: Action schedule
            phi1: Premium
            phi2_values: Indemnity values
            choice_probs: Choice probabilities for this insurer
            
        Returns:
            Expected profit
        """
        if insurer_id == 1:
            delta = self.delta1
        else:
            delta = self.delta2
        
        expected_profit = 0.0
        
        for i, theta in enumerate(self.theta_grid):
            prob_choice = choice_probs[i]
            
            # Expected indemnity payment
            p_action = self.functions.p(a_schedule[i], self.params)  # Probability of no accident
            state_space = self.functions.f(a_schedule[i], delta, self.params)
            _, z_probs = state_space.get_all_states()
            expected_indemnity = np.sum(phi2_values * z_probs) * (1 - p_action)
            
            # Profit per customer: premium - expected indemnity (no monitoring cost per customer)
            profit_per_customer = phi1 - expected_indemnity - self.functions.c(delta, self.params)
            
            # Weight by choice probability and type distribution
            expected_profit += prob_choice * profit_per_customer * self.h_theta[i]
        
        return expected_profit * self.N  # Total expected profit
    
    def grid_search_contracts(self, insurer_id: int, phi1_grid, phi2_grid, logger=None):
        """
        Brute-force grid search for all feasible contracts for one insurer.
        Returns a list of dicts: {'phi1': ..., 'phi2': ..., 'a_schedule': ..., 'utilities': ..., ...}
        """
        if insurer_id == 1:
            delta = self.delta1
        else:
            delta = self.delta2
        contracts = []
        for phi1 in phi1_grid:
            for phi2_tuple in phi2_grid:
                phi2_values = np.array(phi2_tuple)
                a_schedule = []
                feasible = True
                for theta in self.theta_grid:
                    # Solve incentive constraint = 0 for a in action bounds
                    def ic_func(a):
                        return self.incentive_constraint(a, phi1, phi2_values, delta, theta)
                    try:
                        sol = root_scalar(ic_func, bracket=(self.a_min, self.a_max), method='brentq')
                        if not sol.converged:
                            # Log convergence failure details
                            print(f"Root finding failed to converge for theta={theta}, "
                                  f"phi1={phi1}, phi2={phi2_tuple}. "
                                  f"Final function value: {sol.function_calls[-1].fval}")
                            feasible = False
                            break
                        a_schedule.append(sol.root)
                    except Exception as e:
                        # Evaluate function at boundaries to understand failure
                        try:
                            f_lower = ic_func(self.a_min)
                            f_upper = ic_func(self.a_max)
                        except Exception as e2:
                            f_lower = None
                            f_upper = None
                        # Log the failure
                        # if logger is not None:
                        #     logger.log_warning(f"Failed to solve incentive constraint for insurer {insurer_id}, theta={theta}, phi1={phi1}, phi2={phi2_tuple}: {e}. Function at bounds: lower={f_lower}, upper={f_upper}")
                        # else:
                        #     print(f"Failed to solve incentive constraint for insurer {insurer_id}, theta={theta}, phi1={phi1}, phi2={phi2_tuple}: {e}. Function at bounds: lower={f_lower}, upper={f_upper}")
                        feasible = False
                        break
                if feasible:
                    a_schedule = np.array(a_schedule)
                    utilities = self.compute_utility_grid(insurer_id, a_schedule, phi1, phi2_values)
                    # Compute expected profit for this contract (single-insurer, so use uniform choice probs)
                    uniform_choice_probs = np.ones(self.n_theta) / self.n_theta
                    expected_profit = self.compute_expected_profit(insurer_id, a_schedule, phi1, phi2_values, uniform_choice_probs)
                    # Add both naming conventions for compatibility
                    contracts.append({
                        'phi1': phi1,
                        'phi2': phi2_values,
                        'a_schedule': a_schedule,
                        'utilities': utilities,
                        'premium': phi1,
                        'indemnity_values': phi2_values,
                        'expected_profit': expected_profit
                    })
        return contracts

    def _process_contract_parallel(self, args):
        """
        Helper function for parallel processing of individual contracts.
        
        Args:
            args: Tuple containing (insurer_id, phi1, phi2_tuple, delta, theta_grid, a_min, a_max, params)
            
        Returns:
            Contract dictionary if feasible, None otherwise
        """
        insurer_id, phi1, phi2_tuple, delta, theta_grid, a_min, a_max, params = args
        
        # Create a temporary solver instance for this worker
        # We need to recreate the functions and solver for multiprocessing
        function_config = {
            'p': 'logistic',
            'm': 'linear', 
            'e': 'power',
            'u': 'logarithmic',
            'f': 'binary_states',
            'c': 'linear'
        }
        functions = FlexibleFunctions(function_config)
        temp_solver = DuopolySolver(functions, params)
        
        phi2_values = np.array(phi2_tuple)
        a_schedule = []
        feasible = True
        
        for theta in theta_grid:
            # Solve incentive constraint = 0 for a in action bounds
            def ic_func(a):
                return temp_solver.incentive_constraint(a, phi1, phi2_values, delta, theta)
            try:
                sol = root_scalar(ic_func, bracket=(a_min, a_max), method='brentq')
                if not sol.converged:
                    feasible = False
                    break
                a_schedule.append(sol.root)
            except Exception as e:
                feasible = False
                break
                
        if feasible:
            a_schedule = np.array(a_schedule)
            utilities = temp_solver.compute_utility_grid(insurer_id, a_schedule, phi1, phi2_values)
            # Compute expected profit for this contract (single-insurer, so use uniform choice probs)
            uniform_choice_probs = np.ones(len(theta_grid)) / len(theta_grid)
            expected_profit = temp_solver.compute_expected_profit(insurer_id, a_schedule, phi1, phi2_values, uniform_choice_probs)
            
            return {
                'phi1': phi1,
                'phi2': phi2_values,
                'a_schedule': a_schedule,
                'utilities': utilities,
                'premium': phi1,
                'indemnity_values': phi2_values,
                'expected_profit': expected_profit
            }
        else:
            return None

    @staticmethod
    def _process_contract_parallel_static(args):
        """
        Static helper function for parallel processing of individual contracts.
        
        Args:
            args: Tuple containing (insurer_id, phi1, phi2_tuple, delta, theta_grid, a_min, a_max, params, function_config)
            
        Returns:
            Contract dictionary if feasible, None otherwise
        """
        insurer_id, phi1, phi2_tuple, delta, theta_grid, a_min, a_max, params, function_config = args
        
        # Create a temporary solver instance for this worker
        functions = FlexibleFunctions(function_config)
        temp_solver = DuopolySolver(functions, params)
        
        phi2_values = np.array(phi2_tuple)
        a_schedule = []
        feasible = True
        
        for theta in theta_grid:
            # Solve incentive constraint = 0 for a in action bounds
            def ic_func(a):
                return temp_solver.incentive_constraint(a, phi1, phi2_values, delta, theta)
            try:
                sol = root_scalar(ic_func, bracket=(a_min, a_max), method='brentq')
                if not sol.converged:
                    feasible = False
                    break
                a_schedule.append(sol.root)
            except Exception as e:
                feasible = False
                break
                
        if feasible:
            a_schedule = np.array(a_schedule)
            utilities = temp_solver.compute_utility_grid(insurer_id, a_schedule, phi1, phi2_values)
            # Compute expected profit for this contract (single-insurer, so use uniform choice probs)
            uniform_choice_probs = np.ones(len(theta_grid)) / len(theta_grid)
            expected_profit = temp_solver.compute_expected_profit(insurer_id, a_schedule, phi1, phi2_values, uniform_choice_probs)
            
            return {
                'phi1': phi1,
                'phi2': phi2_values,
                'a_schedule': a_schedule,
                'utilities': utilities,
                'premium': phi1,
                'indemnity_values': phi2_values,
                'expected_profit': expected_profit
            }
        else:
            return None

    def grid_search_contracts_parallel(self, insurer_id: int, phi1_grid, phi2_grid, logger=None, n_jobs=None):
        """
        Parallel brute-force grid search for all feasible contracts for one insurer.
        
        Args:
            insurer_id: 1 or 2 for the insurer
            phi1_grid: Grid of premium values
            phi2_grid: Grid of indemnity value tuples
            logger: Optional logger instance
            n_jobs: Number of parallel jobs (default: number of CPU cores)
            
        Returns:
            List of dicts: {'phi1': ..., 'phi2': ..., 'a_schedule': ..., 'utilities': ..., ...}
        """
        if insurer_id == 1:
            delta = self.delta1
        else:
            delta = self.delta2
            
        if n_jobs is None:
            n_jobs = mp.cpu_count()
            
        print(f"Starting parallel grid search for insurer {insurer_id} with {n_jobs} workers...")
        start_time = time.time()
        
        # Get the function configuration from the current solver
        function_config = self.get_function_config()
        
        # Prepare arguments for parallel processing
        args_list = []
        for phi1 in phi1_grid:
            for phi2_tuple in phi2_grid:
                args = (insurer_id, phi1, phi2_tuple, delta, self.theta_grid, 
                       self.a_min, self.a_max, self.params, function_config)
                args_list.append(args)
        
        # Use multiprocessing to process contracts in parallel
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(DuopolySolver._process_contract_parallel_static, args_list)
        
        # Filter out None results (infeasible contracts)
        contracts = [contract for contract in results if contract is not None]
        
        end_time = time.time()
        print(f"Parallel grid search completed for insurer {insurer_id} in {end_time - start_time:.2f} seconds")
        print(f"Found {len(contracts)} feasible contracts out of {len(args_list)} total combinations")
        
        return contracts

    def get_function_config(self):
        """Get the function configuration from the current solver."""
        return self.functions.config

    def _process_contract_pair_parallel(self, args):
        """
        Helper function for parallel processing of contract pairs.
        
        Args:
            args: Tuple containing (c1, c2, V0, mu, params)
            
        Returns:
            Result dictionary for the contract pair
        """
        c1, c2, V0, mu, params = args
        
        # Create temporary solver for this worker
        function_config = {
            'p': 'logistic',
            'm': 'linear', 
            'e': 'power',
            'u': 'logarithmic',
            'f': 'binary_states',
            'c': 'linear'
        }
        functions = FlexibleFunctions(function_config)
        temp_solver = DuopolySolver(functions, params)
        
        V1 = c1['utilities']
        V2 = c2['utilities']
        
        # Logit choice probabilities
        P0, P1, P2 = FlexibleFunctions.choice_probabilities(V0, V1, V2, mu)
        profit_1 = temp_solver.compute_expected_profit(1, c1['a_schedule'], c1['phi1'], c1['phi2'], P1)
        profit_2 = temp_solver.compute_expected_profit(2, c2['a_schedule'], c2['phi1'], c2['phi2'], P2)
        
        # Add profit_1, profit_2, and expected_profit to insurer dicts for downstream compatibility
        c1_copy = dict(c1)  # Copy to avoid mutating original
        c2_copy = dict(c2)
        c1_copy['profit_1'] = profit_1
        c1_copy['expected_profit'] = profit_1
        c2_copy['profit_2'] = profit_2
        c2_copy['expected_profit'] = profit_2
        
        return {
            'insurer1': c1_copy,
            'insurer2': c2_copy,
            'choice_probabilities': {'P0': P0, 'P1': P1, 'P2': P2},
            'utilities': {'V0': V0, 'V1': V1, 'V2': V2},
            'profit_1': profit_1,
            'profit_2': profit_2
        }

    @staticmethod
    def _process_contract_pair_parallel_static(args):
        """
        Static helper function for parallel processing of contract pairs.
        
        Args:
            args: Tuple containing (c1, c2, V0, mu, params, function_config)
            
        Returns:
            Result dictionary for the contract pair
        """
        c1, c2, V0, mu, params, function_config = args
        
        # Create temporary solver for this worker
        functions = FlexibleFunctions(function_config)
        temp_solver = DuopolySolver(functions, params)
        
        V1 = c1['utilities']
        V2 = c2['utilities']
        
        # Logit choice probabilities
        P0, P1, P2 = FlexibleFunctions.choice_probabilities(V0, V1, V2, mu)
        profit_1 = temp_solver.compute_expected_profit(1, c1['a_schedule'], c1['phi1'], c1['phi2'], P1)
        profit_2 = temp_solver.compute_expected_profit(2, c2['a_schedule'], c2['phi1'], c2['phi2'], P2)
        
        # Add profit_1, profit_2, and expected_profit to insurer dicts for downstream compatibility
        c1_copy = dict(c1)  # Copy to avoid mutating original
        c2_copy = dict(c2)
        c1_copy['profit_1'] = profit_1
        c1_copy['expected_profit'] = profit_1
        c2_copy['profit_2'] = profit_2
        c2_copy['expected_profit'] = profit_2
        
        return {
            'insurer1': c1_copy,
            'insurer2': c2_copy,
            'choice_probabilities': {'P0': P0, 'P1': P1, 'P2': P2},
            'utilities': {'V0': V0, 'V1': V1, 'V2': V2},
            'profit_1': profit_1,
            'profit_2': profit_2
        }

    def brute_force_duopoly(self, logger=None, n_jobs=None):
        """
        Parallel brute-force grid search for both insurers, find all Pareto optimal equilibria.
        
        Args:
            logger: Optional logger instance
            n_jobs: Number of parallel jobs (default: number of CPU cores)
            
        Returns:
            List of Pareto optimal contract pairs and their outcomes
        """
        if n_jobs is None:
            n_jobs = mp.cpu_count()
            
        print(f"Starting parallel duopoly grid search with {n_jobs} workers...")
        start_time = time.time()
        
        # Get grid parameters
        n_phi1 = int(self.params.get('n_phi1_grid', 50))
        n_phi2 = int(self.params.get('n_phi2_grid', 50))
        phi1_grid = np.linspace(0.1, self.s, n_phi1)
        phi2_grid_1d = np.linspace(0.0, self.s, n_phi2)
        
        # Cartesian product for phi2(z) for all states
        phi2_grid = list(product(phi2_grid_1d, repeat=self.n_states))
        
        # Get the function configuration from the current solver
        function_config = self.get_function_config()
        
        # Enumerate all contracts for each insurer in parallel
        print("Computing contracts for insurer 1...")
        contracts_1 = self.grid_search_contracts_parallel(1, phi1_grid, phi2_grid, logger=logger, n_jobs=n_jobs)
        
        print("Computing contracts for insurer 2...")
        contracts_2 = self.grid_search_contracts_parallel(2, phi1_grid, phi2_grid, logger=logger, n_jobs=n_jobs)
        
        # Evaluate all contract pairs in parallel
        print("Evaluating contract pairs...")
        V0 = self.compute_reservation_utility_grid()
        mu = self.params.get('xi_scale')
        
        # Prepare arguments for parallel processing of contract pairs
        args_list = []
        for c1 in contracts_1:
            for c2 in contracts_2:
                args = (c1, c2, V0, mu, self.params, function_config)
                args_list.append(args)
        
        # Use multiprocessing to process contract pairs in parallel
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(DuopolySolver._process_contract_pair_parallel_static, args_list)
        
        # Find Pareto optimal contract pairs
        pareto = self.pareto_optimal_solutions(results)
        
        end_time = time.time()
        print(f"Parallel duopoly grid search completed in {end_time - start_time:.2f} seconds")
        print(f"Evaluated {len(results)} contract pairs, found {len(pareto)} Pareto optimal solutions")
        
        return pareto

    @staticmethod
    def pareto_optimal_solutions(results):
        """
        Given a list of contract pair results, return the Pareto optimal subset.
        Each result should be a dict with 'profit_1' and 'profit_2'.
        """
        pareto = []
        for i, res_i in enumerate(results):
            dominated = False
            for j, res_j in enumerate(results):
                if i == j:
                    continue
                # res_j dominates res_i if both profits are >= and at least one is strictly >
                if (res_j['profit_1'] >= res_i['profit_1'] and
                    res_j['profit_2'] >= res_i['profit_2'] and
                    (res_j['profit_1'] > res_i['profit_1'] or res_j['profit_2'] > res_i['profit_2'])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(res_i)
        return pareto



# ============================================================================
# SIMULATION & PLOTTING
# ============================================================================

class InsuranceSimulator:
    """Simulation and plotting class for the insurance model."""
    
    def __init__(self, solver: DuopolySolver):
        """
        Initialize the simulator.
        
        Args:
            solver: DuopolySolver instance
        """
        self.solver = solver
    
    def run_simulation(self) -> Dict:
        """
        Run the full simulation.
        
        Returns:
            Dictionary containing simulation results
        """
        print("Starting duopoly insurance simulation...")
        solution = self.solver.brute_force_duopoly()
        print("Simulation completed!")
        return solution
    
    def plot_results(self, solution: Dict, save_path: str = None):
        """
        Plot simulation results including choice probabilities.
        
        Args:
            solution: Solution dictionary from solver
            save_path: Optional path to save plots
        """
        # Use the first Pareto optimal solution for plotting
        if isinstance(solution, list):
            solution = solution[0]
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Plot 1: Action schedules
        theta_grid = self.solver.theta_grid
        a1 = solution['insurer1']['a_schedule']
        a2 = solution['insurer2']['a_schedule']
        
        axes[0, 0].plot(theta_grid, a1, 'b-', linewidth=2, label='Insurer 1')
        axes[0, 0].plot(theta_grid, a2, 'r--', linewidth=2, label='Insurer 2')
        axes[0, 0].set_xlabel(r'$\theta$ (Risk Type)')
        axes[0, 0].set_ylabel(r'$a^i(\theta)$ (Action Level)')
        axes[0, 0].set_title('Optimal Action Schedules')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Premiums
        phi1_1 = solution['insurer1']['phi1']
        phi1_2 = solution['insurer2']['phi1']
        
        axes[0, 1].bar(['Insurer 1', 'Insurer 2'], [phi1_1, phi1_2], 
                      color=['blue', 'red'], alpha=0.7)
        axes[0, 1].set_ylabel(r'$\phi_1^i$ (Premium)')
        axes[0, 1].set_title('Optimal Premiums')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Choice probabilities
        P0 = solution['choice_probabilities']['P0']
        P1 = solution['choice_probabilities']['P1']
        P2 = solution['choice_probabilities']['P2']
        
        axes[1, 0].plot(theta_grid, P0, 'g-', linewidth=2, label='No Insurance')
        axes[1, 0].plot(theta_grid, P1, 'b-', linewidth=2, label='Insurer 1')
        axes[1, 0].plot(theta_grid, P2, 'r-', linewidth=2, label='Insurer 2')
        axes[1, 0].set_xlabel(r'$\theta$ (Risk Type)')
        axes[1, 0].set_ylabel('Choice Probability')
        axes[1, 0].set_title('Choice Probabilities by Risk Type')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Expected profits
        profit_1 = solution['insurer1']['profit_1']
        profit_2 = solution['insurer2']['profit_2']
        
        axes[1, 1].bar(['Insurer 1', 'Insurer 2'], [profit_1, profit_2], 
                      color=['blue', 'red'], alpha=0.7)
        axes[1, 1].set_ylabel('Expected Profit')
        axes[1, 1].set_title('Expected Profits')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Discrete indemnity schedules
        z_values = self.solver.z_values
        phi2_1 = solution['insurer1']['phi2']
        phi2_2 = solution['insurer2']['phi2']
        
        # Create bar plot for discrete indemnity values
        x_pos = np.arange(len(z_values))
        width = 0.35
        
        axes[2, 0].bar(x_pos - width/2, phi2_1, width, label='Insurer 1', alpha=0.7, color='blue')
        axes[2, 0].bar(x_pos + width/2, phi2_2, width, label='Insurer 2', alpha=0.7, color='red')
        axes[2, 0].set_xlabel('State Index')
        axes[2, 0].set_ylabel(r'$\phi_2^i(z)$ (Indemnity)')
        axes[2, 0].set_title('Discrete Indemnity Schedules')
        axes[2, 0].set_xticks(x_pos)
        axes[2, 0].set_xticklabels([f'z={z:.1f}' for z in z_values])
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Utilities comparison
        V0 = solution['utilities']['V0']
        V1 = solution['utilities']['V1']
        V2 = solution['utilities']['V2']
        
        axes[2, 1].plot(theta_grid, V0, 'g-', linewidth=2, label='No Insurance')
        axes[2, 1].plot(theta_grid, V1, 'b-', linewidth=2, label='Insurer 1')
        axes[2, 1].plot(theta_grid, V2, 'r-', linewidth=2, label='Insurer 2')
        axes[2, 1].set_xlabel(r'$\theta$ (Risk Type)')
        axes[2, 1].set_ylabel('Expected Utility')
        axes[2, 1].set_title('Utilities by Risk Type')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_discrete_analysis(self, solution: Dict, save_path: str = None):
        """
        Additional plots specifically for discrete state space analysis.
        
        Args:
            solution: Solution dictionary from solver
            save_path: Optional path to save plots
        """
        # Use the first Pareto optimal solution for plotting
        if isinstance(solution, list):
            solution = solution[0]
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        z_values = self.solver.z_values
        z_probs = self.solver.z_probs
        phi2_1 = solution['insurer1']['phi2']
        phi2_2 = solution['insurer2']['phi2']
        
        # Plot 1: State probabilities
        axes[0, 0].bar(range(len(z_values)), z_probs, alpha=0.7, color='green')
        axes[0, 0].set_xlabel('State Index')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].set_title('State Space Probabilities')
        axes[0, 0].set_xticks(range(len(z_values)))
        axes[0, 0].set_xticklabels([f'z={z:.1f}' for z in z_values])
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Indemnity vs State Value
        axes[0, 1].scatter(z_values, phi2_1, s=100, alpha=0.7, label='Insurer 1', color='blue')
        axes[0, 1].scatter(z_values, phi2_2, s=100, alpha=0.7, label='Insurer 2', color='red')
        axes[0, 1].set_xlabel('State Value (z)')
        axes[0, 1].set_ylabel('Indemnity')
        axes[0, 1].set_title('Indemnity vs State Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Expected indemnity by risk type
        theta_grid = self.solver.theta_grid
        a1 = solution['insurer1']['a_schedule']
        a2 = solution['insurer2']['a_schedule']
        
        expected_indemnity_1 = []
        expected_indemnity_2 = []
        
        for i, theta in enumerate(theta_grid):
            # Get state space for this action level
            state_space_1 = self.solver.functions.f(a1[i], self.solver.delta1, self.solver.params)
            _, z_probs_1 = state_space_1.get_all_states()
            
            state_space_2 = self.solver.functions.f(a2[i], self.solver.delta2, self.solver.params)
            _, z_probs_2 = state_space_2.get_all_states()
            
            # Expected indemnity
            exp_ind_1 = np.sum(phi2_1 * z_probs_1)
            exp_ind_2 = np.sum(phi2_2 * z_probs_2)
            
            expected_indemnity_1.append(exp_ind_1)
            expected_indemnity_2.append(exp_ind_2)
        
        axes[1, 0].plot(theta_grid, expected_indemnity_1, 'b-', linewidth=2, label='Insurer 1')
        axes[1, 0].plot(theta_grid, expected_indemnity_2, 'r--', linewidth=2, label='Insurer 2')
        axes[1, 0].set_xlabel(r'$\theta$ (Risk Type)')
        axes[1, 0].set_ylabel('Expected Indemnity')
        axes[1, 0].set_title('Expected Indemnity by Risk Type')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Contract comparison
        contract_data = [
            ['Premium', solution['insurer1']['phi1'], solution['insurer2']['phi1']],
            ['Avg Action', np.mean(a1), np.mean(a2)],
            ['Avg Indemnity', np.mean(phi2_1), np.mean(phi2_2)],
            ['Max Indemnity', np.max(phi2_1), np.max(phi2_2)]
        ]
        
        labels = [row[0] for row in contract_data]
        insurer1_vals = [row[1] for row in contract_data]
        insurer2_vals = [row[2] for row in contract_data]
        
        x_pos = np.arange(len(labels))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, insurer1_vals, width, label='Insurer 1', alpha=0.7, color='blue')
        axes[1, 1].bar(x_pos + width/2, insurer2_vals, width, label='Insurer 2', alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Contract Features')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Contract Comparison')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(labels)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def sensitivity_analysis(self, param_name: str, param_range: List[float], 
                           save_path: str = None):
        """
        Perform sensitivity analysis by varying a parameter.
        
        Args:
            param_name: Name of parameter to vary (required)
            param_range: List of parameter values to test (required)
            save_path: Optional path to save plots
        """
        if param_name is None:
            raise ValueError("param_name is required")
        if param_range is None:
            raise ValueError("param_range is required")
        if param_name not in self.solver.params:
            raise ValueError(f"Parameter '{param_name}' not found in solver parameters")
        
        results = []
        
        for param_val in param_range:
            # Update parameter
            original_val = self.solver.params[param_name]
            self.solver.params[param_name] = param_val
            
            # Solve
            solution = self.solver.brute_force_duopoly()
            results.append({
                'param_val': param_val,
                'premium_1': solution['insurer1']['phi1'],
                'premium_2': solution['insurer2']['phi1'],
                'avg_action_1': np.mean(solution['insurer1']['a_schedule']),
                'avg_action_2': np.mean(solution['insurer2']['a_schedule']),
                'avg_indemnity_1': np.mean(solution['insurer1']['phi2']),
                'avg_indemnity_2': np.mean(solution['insurer2']['phi2'])
            })
            
            # Restore original parameter
            self.solver.params[param_name] = original_val
        
        # Plot sensitivity results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        param_vals = [r['param_val'] for r in results]
        
        axes[0, 0].plot(param_vals, [r['premium_1'] for r in results], 'b-', linewidth=2, label='Insurer 1')
        axes[0, 0].plot(param_vals, [r['premium_2'] for r in results], 'r--', linewidth=2, label='Insurer 2')
        axes[0, 0].set_xlabel(param_name)
        axes[0, 0].set_ylabel('Premium')
        axes[0, 0].set_title(f'Premium Sensitivity to {param_name}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(param_vals, [r['avg_action_1'] for r in results], 'b-', linewidth=2, label='Insurer 1')
        axes[0, 1].plot(param_vals, [r['avg_action_2'] for r in results], 'r--', linewidth=2, label='Insurer 2')
        axes[0, 1].set_xlabel(param_name)
        axes[0, 1].set_ylabel('Average Action Level')
        axes[0, 1].set_title(f'Action Level Sensitivity to {param_name}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(param_vals, [r['avg_indemnity_1'] for r in results], 'b-', linewidth=2, label='Insurer 1')
        axes[1, 0].plot(param_vals, [r['avg_indemnity_2'] for r in results], 'r--', linewidth=2, label='Insurer 2')
        axes[1, 0].set_xlabel(param_name)
        axes[1, 0].set_ylabel('Average Indemnity')
        axes[1, 0].set_title(f'Indemnity Sensitivity to {param_name}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def run_simulation(state_spaces=None, include_sensitivity=True, save_plots=True, params=None, logger: SimulationLogger = None, use_parallel=True, n_jobs=None):
    """
    Unified simulation function that can handle single or multiple state space configurations.
    
    Args:
        state_spaces: List of state space configurations to test. 
                     If None, uses default binary states.
                     Each config should be {'name': str, 'f': str} where 'f' is the state density function name.
        include_sensitivity: Whether to run sensitivity analysis
        save_plots: Whether to save plots to files
        params: Dictionary containing all required parameters. Must include:
               - W: Initial wealth
               - s: Accident severity
               - N: Number of customers
               - delta1: Insurer 1 monitoring level
               - delta2: Insurer 2 monitoring level
               - theta_min: Minimum risk type
               - theta_max: Maximum risk type
               - n_theta: Number of risk types
               - a_min: Minimum action level (default: 0.0)
               - a_max: Maximum action level (default: 1.0)
               - p_alpha: Parameter for no accident probability function
               - p_beta: Parameter for no accident probability function
               - m_gamma: Parameter for monitoring cost function
               - e_kappa: Parameter for action cost function
               - e_power: Parameter for action cost function
               - u_rho: Parameter for utility function
               - f_p_base: Parameter for state density function
               - c_lambda: Parameter for insurer cost function
               - xi_scale: Scale parameter for logit choice model
        logger: Optional SimulationLogger instance for logging
        use_parallel: Whether to use parallel processing (default: True)
        n_jobs: Number of parallel jobs (default: number of CPU cores)
    Returns:
        Dictionary containing simulation results
    """
    # Validate required parameters
    if params is None:
        raise ValueError("params dictionary is required and cannot be None")
    required_params = [
        'W', 's', 'N', 'delta1', 'delta2', 'theta_min', 'theta_max', 'n_theta',
        'p_alpha', 'p_beta', 'm_gamma', 'e_kappa', 'e_power', 'u_rho', 
        'f_p_base', 'c_lambda', 'xi_scale'
    ]
    missing_params = [param for param in required_params if param not in params]
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")
    
    # Check action bounds (optional parameters with defaults)
    if 'a_min' not in params:
        params['a_min'] = 0.0
    if 'a_max' not in params:
        params['a_max'] = 1.0
    
    # Validate action bounds
    if params['a_min'] >= params['a_max']:
        raise ValueError("a_min must be less than a_max")
    if params['a_min'] < 0:
        raise ValueError("a_min must be non-negative")
    
    for state_config in state_spaces:
        if state_config['f'] == 'custom_discrete':
            custom_params = ['f_z_values', 'f_base_probs', 'f_p_a', 'f_p_delta']
            missing_custom = [param for param in custom_params if param not in params]
            if missing_custom:
                raise ValueError(f"Missing required parameters for custom_discrete: {missing_custom}")
    if state_spaces is None:
        state_spaces = [{'name': 'binary', 'f': 'binary_states'}]
    base_function_config = {
        'p': 'logistic',
        'm': 'linear',
        'e': 'power',
        'u': 'logarithmic',
        'c': 'linear'
    }
    results = {}
    # --- Initialize logger if not provided ---
    if logger is None:
        logger = SimulationLogger(experiment_name="duopoly_experiment")
    logger.log_experiment_start("Duopoly insurance simulation")
    logger.log_simulation_settings({
        'state_spaces': state_spaces,
        'include_sensitivity': include_sensitivity,
        'save_plots': save_plots,
        'use_parallel': use_parallel,
        'n_jobs': n_jobs
    })
    logger.log_parameters(params)
    # ------------------------------------------------------------------
    # Ensure files from different simulation runs are stored separately.
    # Each run gets its own sub-directory named with the logger's
    # experiment name and a timestamp so that earlier results are never
    # overwritten.
    # ------------------------------------------------------------------
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('outputs') / f"{logger.experiment_name}_{timestamp_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for state_config in state_spaces:
        print(f"\nRunning simulation with {state_config['name']} state space...")
        function_config = base_function_config.copy()
        function_config['f'] = state_config['f']
        functions = FlexibleFunctions(function_config)
        solver = DuopolySolver(functions, params)
        
        # Choose between parallel and sequential grid search
        if use_parallel:
            print("Using parallel brute-force grid search for contracts...")
            pareto_solutions = solver.brute_force_duopoly(logger=logger, n_jobs=n_jobs)
        else:
            print("Using sequential brute-force grid search for contracts...")
            pareto_solutions = solver.brute_force_duopoly(logger=logger)
            
        results[state_config['name']] = pareto_solutions
        if save_plots:
            # Only plot the first Pareto solution for illustration
            if pareto_solutions:
                solution = pareto_solutions[0]
                simulator = InsuranceSimulator(solver)
                simulator.plot_results(solution, str(output_dir / f'{state_config["name"]}_results.png'))
                simulator.plot_discrete_analysis(solution, str(output_dir / f'{state_config["name"]}_analysis.png'))
        # Log duopoly solution
        if pareto_solutions:
            logger.log_duopoly_solution(pareto_solutions[0], state_config['name'])
        else:
            logger.log_warning(f"No Pareto solutions found for state space: {state_config['name']}")
        # Sensitivity analysis can be skipped or adapted for grid search
    print("Simulation completed!")
    logger.log_experiment_end("Simulation completed successfully")
    logger.print_summary()
    return results


if __name__ == "__main__":
    # Example: Multiple state spaces with all required parameters
    print("\n" + "=" * 60)
    print("EXAMPLE")
    print("=" * 60)
    required_params = {
        'W': 5000.0,           # Initial wealth
        's': 800.0,            # Accident severity
        'N': 10000,             # Number of customers
        'delta1': 1.0,          # Insurer 1 monitoring level
        'delta2': 0.5,          # Insurer 2 monitoring level
        'theta_min': 0.5,       # Minimum risk type
        'theta_max': 2.0,       # Maximum risk type
        'n_theta': 20,          # Number of risk types
        'a_min': 0.0,           # Minimum action level
        'a_max': 1.0,           # Maximum action level
        'p_alpha': 0,           # Parameter for no accident probability function
        'p_beta': 1.0,          # Parameter for no accident probability function
        'm_gamma': 120,         # Parameter for monitoring cost function
        'e_kappa': 80,          # Parameter for action cost function
        'e_power': 2.0,         # Parameter for action cost function
        'u_rho': 0.1,           # Parameter for utility function
        'f_p_base': 0.5,        # Parameter for state density function
        'c_lambda': 30,         # Parameter for insurer cost function
        'xi_scale': 500.0,      # Scale parameter for logit choice model
        # Additional parameters for custom_discrete state density function
        'f_z_values': [0.0, 1.0, 2.0],  # State values for custom discrete
        'f_base_probs': [0.5, 0.3, 0.2],  # Base probabilities for custom discrete
        'f_p_a': 0.1,           # Action effect parameter for custom discrete
        'f_p_delta': 0.05,      # Monitoring effect parameter for custom discrete
        # Grid search parameters (smaller for faster testing)
        'n_phi1_grid': 20,      # Number of premium grid points
        'n_phi2_grid': 10       # Number of indemnity grid points
    }
    
    # Test action bounds functionality
    print("\nTesting action bounds functionality...")
    function_config = {'p': 'logistic', 'm': 'linear', 'e': 'power', 'u': 'logarithmic', 'f': 'binary_states', 'c': 'linear'}
    functions = FlexibleFunctions(function_config)
    
    # Test with default bounds [0, 1]
    solver_default = DuopolySolver(functions, required_params)
    print(f"Default action bounds: [{solver_default.a_min}, {solver_default.a_max}]")
    
    # Test with custom bounds [0.2, 0.8]
    custom_params = required_params.copy()
    custom_params['a_min'] = 0.2
    custom_params['a_max'] = 0.8
    solver_custom = DuopolySolver(functions, custom_params)
    print(f"Custom action bounds: [{solver_custom.a_min}, {solver_custom.a_max}]")
    
    # Test reservation utility computation with different bounds
    theta_test = 1.0
    V0_default = solver_default.compute_reservation_utility(theta_test)
    V0_custom = solver_custom.compute_reservation_utility(theta_test)
    print(f"Reservation utility with default bounds: {V0_default:.4f}")
    print(f"Reservation utility with custom bounds: {V0_custom:.4f}")
    
    # Initialize logger
    logger = SimulationLogger(experiment_name="duopoly_example_run")
    
    # Run simulation with parallel processing
    print("\n" + "=" * 60)
    print("RUNNING SIMULATION WITH PARALLEL PROCESSING")
    print("=" * 60)
    
    results = run_simulation(
        state_spaces=[
            {'name': 'binary', 'f': 'binary_states'},
            {'name': 'custom_discrete', 'f': 'custom_discrete'}
        ],
        include_sensitivity=False,
        params=required_params,
        logger=logger,
        use_parallel=True,
        n_jobs=None  # Use all available cores
    )
    
    print("\n" + "=" * 60)
    print("Simulations completed! Check the generated plots and logs.")
    print("=" * 60)
