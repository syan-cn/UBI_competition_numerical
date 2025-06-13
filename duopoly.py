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
        """Exponential: u(x) = max * (1 - exp(-rho * x))."""
        if 'u_rho' not in params:
            raise ValueError("Parameter 'u_rho' is required for exponential utility function")
        if 'u_max' not in params:
            raise ValueError("Parameter 'u_max' is required for exponential utility function")
        
        rho = params['u_rho']
        max_val = params['u_max']
        result = max_val * (1 - np.exp(-rho * x))
        return result
    
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
        if 'p' not in self.config:
            raise ValueError("Function type 'p' not specified in configuration")
        func_type = self.config['p']
        return self.p_functions[func_type](a, params)
    
    def m(self, delta: float, params: Dict) -> float:
        """Monitoring cost function."""
        if 'm' not in self.config:
            raise ValueError("Function type 'm' not specified in configuration")
        func_type = self.config['m']
        return self.m_functions[func_type](delta, params)
    
    def e(self, a: float, theta: float, params: Dict) -> float:
        """Action cost function."""
        if 'e' not in self.config:
            raise ValueError("Function type 'e' not specified in configuration")
        func_type = self.config['e']
        return self.e_functions[func_type](a, theta, params)
    
    def u(self, x: float, params: Dict) -> float:
        """Utility function."""
        if 'u' not in self.config:
            raise ValueError("Function type 'u' not specified in configuration")
        func_type = self.config['u']
        return self.u_functions[func_type](x, params)
    
    def f(self, a: float, delta: float, params: Dict) -> DiscreteStateSpace:
        """State density function - now returns DiscreteStateSpace."""
        if 'f' not in self.config:
            raise ValueError("Function type 'f' not specified in configuration")
        func_type = self.config['f']
        return self.f_functions[func_type](a, delta, params)
    
    def c(self, delta: float, params: Dict) -> float:
        """Insurer cost function."""
        if 'c' not in self.config:
            raise ValueError("Function type 'c' not specified in configuration")
        func_type = self.config['c']
        return self.c_functions[func_type](delta, params)

    def dp_da(self, a: float, params: Dict) -> float:
        if 'p' not in self.config:
            raise ValueError("Function type 'p' not specified in configuration")
        func_type = self.config['p']
        return getattr(NoAccidentProbability, f"dp_da_{func_type}")(a, params)

    def de_da(self, a: float, theta: float, params: Dict) -> float:
        if 'e' not in self.config:
            raise ValueError("Function type 'e' not specified in configuration")
        func_type = self.config['e']
        return getattr(ActionCost, f"de_da_{func_type}")(a, theta, params)

    def df_da(self, a: float, delta: float, params: Dict) -> np.ndarray:
        if 'f' not in self.config:
            raise ValueError("Function type 'f' not specified in configuration")
        func_type = self.config['f']
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

    def brute_force_duopoly(self, logger=None, n_jobs=None, evaluation_method='simple'):
        """
        Parallel brute-force grid search for both insurers, find all Pareto optimal equilibria.
        
        Args:
            logger: Optional logger instance
            n_jobs: Number of parallel jobs (default: number of CPU cores)
            evaluation_method: Method for contract pair evaluation:
                              'original' - Evaluate all pairs at once
                              'simple' - Simple efficient method with pre-filtering
                              'incremental' - Incremental Pareto method with chunked processing
                              'divide_conquer' - Divide-and-conquer parallel evaluation
            
        Returns:
            List of Pareto optimal contract pairs and their outcomes
        """
        if n_jobs is None:
            n_jobs = mp.cpu_count()
            
        print(f"Starting parallel duopoly grid search with {n_jobs} workers using {evaluation_method} method...")
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
        
        print(f"Generated {len(contracts_1)} contracts for insurer 1, {len(contracts_2)} for insurer 2")
        
        # Evaluate contract pairs using selected method
        V0 = self.compute_reservation_utility_grid()
        mu = self.params.get('xi_scale')
        
        if evaluation_method == 'original':
            pareto = self._evaluate_all_pairs_original(contracts_1, contracts_2, V0, mu, function_config, n_jobs)
        elif evaluation_method == 'simple':
            pareto = self.simple_efficient_contract_evaluation(contracts_1, contracts_2, V0, mu, n_jobs=n_jobs)
        elif evaluation_method == 'incremental':
            pareto = self.incremental_pareto_evaluation(contracts_1, contracts_2, V0, mu, n_jobs=n_jobs)
        elif evaluation_method == 'divide_conquer':
            pareto = self.divide_conquer_evaluation(contracts_1, contracts_2, V0, mu, n_jobs=n_jobs)
        else:
            raise ValueError(f"Unknown evaluation method: {evaluation_method}")
        
        end_time = time.time()
        print(f"Parallel duopoly grid search completed in {end_time - start_time:.2f} seconds")
        print(f"Found {len(pareto)} Pareto optimal solutions using {evaluation_method} method")
        
        return pareto

    def _evaluate_all_pairs_original(self, contracts_1, contracts_2, V0, mu, function_config, n_jobs):
        """Original method: evaluate all contract pairs at once."""
        print("Evaluating all contract pairs at once...")
        
        # Prepare arguments for parallel processing of contract pairs
        args_list = []
        for c1 in contracts_1:
            for c2 in contracts_2:
                args = (c1, c2, V0, mu, self.params, function_config)
                args_list.append(args)
        
        print(f"Evaluating {len(args_list)} contract pairs...")
        
        # Use multiprocessing to process contract pairs in parallel
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(DuopolySolver._process_contract_pair_parallel_static, args_list)
        
        # Find Pareto optimal contract pairs
        pareto = self.pareto_optimal_solutions(results)
        return pareto

    def simple_efficient_contract_evaluation(self, contracts_1, contracts_2, V0, mu, n_jobs=None):
        """
        Simple efficient method with pre-filtering and optimized batching.
        
        Args:
            contracts_1: List of contracts for insurer 1
            contracts_2: List of contracts for insurer 2
            V0: Reservation utility grid
            mu: Logit scale parameter
            n_jobs: Number of parallel jobs
            
        Returns:
            List of Pareto optimal contract pairs
        """
        if n_jobs is None:
            n_jobs = mp.cpu_count()
            
        print("Using simple efficient contract evaluation...")
        start_time = time.time()
        
        # Step 1: Pre-filter contracts to remove clearly dominated ones
        print("Pre-filtering contracts...")
        filtered_1 = self._prefilter_contracts(contracts_1)
        filtered_2 = self._prefilter_contracts(contracts_2)
        
        print(f"After pre-filtering: {len(filtered_1)} contracts for insurer 1, {len(filtered_2)} for insurer 2")
        reduction_1 = (1 - len(filtered_1) / len(contracts_1)) * 100
        reduction_2 = (1 - len(filtered_2) / len(contracts_2)) * 100
        print(f"Reduced contract space by {reduction_1:.1f}% and {reduction_2:.1f}% respectively")
        
        # Step 2: Evaluate contract pairs with optimized batching
        function_config = self.get_function_config()
        batch_size = min(1000, len(filtered_1) * len(filtered_2) // (n_jobs * 4))  # Adaptive batch size
        
        all_pairs = [(c1, c2) for c1 in filtered_1 for c2 in filtered_2]
        print(f"Evaluating {len(all_pairs)} filtered contract pairs in batches of {batch_size}...")
        
        all_results = []
        for i in range(0, len(all_pairs), batch_size):
            batch = all_pairs[i:i + batch_size]
            args_list = [(c1, c2, V0, mu, self.params, function_config) for c1, c2 in batch]
            
            with mp.Pool(processes=n_jobs) as pool:
                batch_results = pool.map(DuopolySolver._process_contract_pair_parallel_static, args_list)
            
            all_results.extend(batch_results)
            
            if (i // batch_size + 1) % 10 == 0:  # Progress indicator
                print(f"Processed {i + len(batch)}/{len(all_pairs)} pairs...")
        
        # Step 3: Find Pareto optimal solutions
        pareto = self.pareto_optimal_solutions(all_results)
        
        end_time = time.time()
        print(f"Simple efficient evaluation completed in {end_time - start_time:.2f} seconds")
        
        return pareto

    def incremental_pareto_evaluation(self, contracts_1, contracts_2, V0, mu, n_jobs=None, chunk_size=500):
        """
        Incremental Pareto method: builds Pareto frontier incrementally by processing contracts in chunks.
        
        Args:
            contracts_1: List of contracts for insurer 1
            contracts_2: List of contracts for insurer 2
            V0: Reservation utility grid
            mu: Logit scale parameter
            n_jobs: Number of parallel jobs
            chunk_size: Size of each processing chunk
            
        Returns:
            List of Pareto optimal contract pairs
        """
        if n_jobs is None:
            n_jobs = mp.cpu_count()
            
        print("Using incremental Pareto evaluation...")
        start_time = time.time()
        
        function_config = self.get_function_config()
        running_pareto = []
        
        # Process contracts in chunks to maintain memory efficiency
        total_pairs = len(contracts_1) * len(contracts_2)
        processed_pairs = 0
        
        for i in range(0, len(contracts_1), chunk_size):
            chunk_1 = contracts_1[i:i + chunk_size]
            
            for j in range(0, len(contracts_2), chunk_size):
                chunk_2 = contracts_2[j:j + chunk_size]
                
                # Evaluate this chunk
                chunk_pairs = [(c1, c2) for c1 in chunk_1 for c2 in chunk_2]
                args_list = [(c1, c2, V0, mu, self.params, function_config) for c1, c2 in chunk_pairs]
                
                with mp.Pool(processes=n_jobs) as pool:
                    chunk_results = pool.map(DuopolySolver._process_contract_pair_parallel_static, args_list)
                
                # Find Pareto optimal solutions in this chunk
                chunk_pareto = self.pareto_optimal_solutions(chunk_results)
                
                # Merge with running Pareto frontier
                combined_results = running_pareto + chunk_pareto
                running_pareto = self.pareto_optimal_solutions(combined_results)
                
                processed_pairs += len(chunk_pairs)
                if processed_pairs % (chunk_size * chunk_size * 5) == 0:  # Progress indicator
                    print(f"Processed {processed_pairs}/{total_pairs} pairs, current Pareto size: {len(running_pareto)}")
        
        end_time = time.time()
        print(f"Incremental Pareto evaluation completed in {end_time - start_time:.2f} seconds")
        print(f"Final Pareto frontier contains {len(running_pareto)} solutions")
        
        return running_pareto

    def divide_conquer_evaluation(self, contracts_1, contracts_2, V0, mu, n_jobs=None, max_depth=3):
        """
        Divide-and-conquer parallel evaluation: recursively partitions contract pairs into subsets,
        evaluates each subset in parallel, and merges Pareto frontiers.
        
        Args:
            contracts_1: List of contracts for insurer 1
            contracts_2: List of contracts for insurer 2
            V0: Reservation utility grid
            mu: Logit scale parameter
            n_jobs: Number of parallel jobs
            max_depth: Maximum recursion depth
            
        Returns:
            List of Pareto optimal contract pairs
        """
        if n_jobs is None:
            n_jobs = mp.cpu_count()
            
        print("Using divide-and-conquer evaluation...")
        start_time = time.time()
        
        function_config = self.get_function_config()
        
        def evaluate_subset(subset_1, subset_2, depth=0):
            """Recursively evaluate a subset of contracts."""
            subset_size = len(subset_1) * len(subset_2)
            
            # Base case: if subset is small enough or max depth reached, evaluate directly
            if subset_size <= 1000 or depth >= max_depth:
                pairs = [(c1, c2) for c1 in subset_1 for c2 in subset_2]
                args_list = [(c1, c2, V0, mu, self.params, function_config) for c1, c2 in pairs]
                
                with mp.Pool(processes=min(n_jobs, len(args_list))) as pool:
                    results = pool.map(DuopolySolver._process_contract_pair_parallel_static, args_list)
                
                return self.pareto_optimal_solutions(results)
            
            # Recursive case: divide the problem
            mid_1 = len(subset_1) // 2
            mid_2 = len(subset_2) // 2
            
            # Create four quadrants
            quadrants = [
                (subset_1[:mid_1], subset_2[:mid_2]),
                (subset_1[:mid_1], subset_2[mid_2:]),
                (subset_1[mid_1:], subset_2[:mid_2]),
                (subset_1[mid_1:], subset_2[mid_2:])
            ]
            
            # Evaluate quadrants in parallel using multiprocessing
            with mp.Pool(processes=min(n_jobs, 4)) as pool:
                quadrant_results = pool.starmap(
                    lambda s1, s2: evaluate_subset(s1, s2, depth + 1),
                    quadrants
                )
            
            # Merge Pareto frontiers from all quadrants
            all_results = []
            for quad_pareto in quadrant_results:
                all_results.extend(quad_pareto)
            
            return self.pareto_optimal_solutions(all_results)
        
        # Sort contracts by profit for better partitioning
        contracts_1_sorted = sorted(contracts_1, key=lambda x: x.get('expected_profit', 0), reverse=True)
        contracts_2_sorted = sorted(contracts_2, key=lambda x: x.get('expected_profit', 0), reverse=True)
        
        pareto = evaluate_subset(contracts_1_sorted, contracts_2_sorted)
        
        end_time = time.time()
        print(f"Divide-and-conquer evaluation completed in {end_time - start_time:.2f} seconds")
        
        return pareto

    def _prefilter_contracts(self, contracts):
        """
        Pre-filter contracts to remove clearly dominated ones.
        
        Args:
            contracts: List of contract dictionaries
            
        Returns:
            List of filtered contracts
        """
        if not contracts:
            return contracts
        
        # Extract key metrics for filtering
        profits = [c.get('expected_profit', 0) for c in contracts]
        premiums = [c.get('phi1', 0) for c in contracts]
        
        # Filter out contracts with very low profits (bottom 25%)
        profit_threshold = np.percentile(profits, 25)
        
        # Filter out contracts with extreme premiums (top/bottom 5%)
        premium_low = np.percentile(premiums, 5)
        premium_high = np.percentile(premiums, 95)
        
        filtered = []
        for i, contract in enumerate(contracts):
            profit = profits[i]
            premium = premiums[i]
            
            # Keep contract if it meets basic criteria
            if (profit >= profit_threshold and 
                premium_low <= premium <= premium_high):
                filtered.append(contract)
        
        # Always keep at least top 50% by profit to avoid over-filtering
        if len(filtered) < len(contracts) * 0.5:
            contracts_by_profit = sorted(contracts, key=lambda x: x.get('expected_profit', 0), reverse=True)
            filtered = contracts_by_profit[:len(contracts) // 2]
        
        return filtered

    def _select_contract_subset(self, contracts_1, contracts_2, existing_pareto, subset_size):
        """
        Select a subset of contracts for evaluation based on existing Pareto solutions.
        
        Args:
            contracts_1: All contracts for insurer 1
            contracts_2: All contracts for insurer 2
            existing_pareto: Existing Pareto optimal solutions
            subset_size: Target subset size
            
        Returns:
            Tuple of (subset_1, subset_2)
        """
        if not existing_pareto:
            # First iteration: random selection
            import random
            subset_1 = random.sample(contracts_1, min(subset_size // 2, len(contracts_1)))
            subset_2 = random.sample(contracts_2, min(subset_size // 2, len(contracts_2)))
            return subset_1, subset_2
        
        # Subsequent iterations: focus on promising regions
        # Extract features from existing Pareto solutions
        pareto_features_1 = self._extract_contract_features([p['insurer1'] for p in existing_pareto])
        pareto_features_2 = self._extract_contract_features([p['insurer2'] for p in existing_pareto])
        
        # Find contracts similar to Pareto solutions
        subset_1 = self._find_similar_contracts(contracts_1, pareto_features_1, subset_size // 2)
        subset_2 = self._find_similar_contracts(contracts_2, pareto_features_2, subset_size // 2)
        
        return subset_1, subset_2

    def _extract_contract_features(self, contracts):
        """
        Extract numerical features from contracts for similarity analysis.
        
        Args:
            contracts: List of contract dictionaries
            
        Returns:
            Array of features
        """
        features = []
        for contract in contracts:
            # Extract key features: premium, average indemnity, action schedule statistics
            phi1 = contract['phi1']
            phi2_mean = np.mean(contract['phi2'])
            phi2_std = np.std(contract['phi2'])
            a_mean = np.mean(contract['a_schedule'])
            a_std = np.std(contract['a_schedule'])
            
            features.append([phi1, phi2_mean, phi2_std, a_mean, a_std])
        
        return np.array(features)

    def _find_similar_contracts(self, contracts, pareto_features, n_contracts):
        """
        Find contracts similar to Pareto solutions using feature similarity.
        
        Args:
            contracts: All available contracts
            pareto_features: Features of Pareto solutions
            n_contracts: Number of contracts to select
            
        Returns:
            List of selected contracts
        """
        if len(contracts) <= n_contracts:
            return contracts
        
        # Extract features from all contracts
        all_features = self._extract_contract_features(contracts)
        
        # Calculate similarity to Pareto solutions
        similarities = []
        for i, contract_features in enumerate(all_features):
            # Minimum distance to any Pareto solution
            distances = np.linalg.norm(pareto_features - contract_features, axis=1)
            min_distance = np.min(distances)
            similarities.append((min_distance, i))
        
        # Select contracts with highest similarity (lowest distance)
        similarities.sort()
        selected_indices = [idx for _, idx in similarities[:n_contracts]]
        
        return [contracts[i] for i in selected_indices]

    def _evaluate_contract_subset_parallel(self, subset_1, subset_2, V0, mu, 
                                         function_config, n_jobs, evaluated_pairs):
        """
        Evaluate a subset of contract pairs in parallel with early termination.
        
        Args:
            subset_1: Subset of contracts for insurer 1
            subset_2: Subset of contracts for insurer 2
            V0: Reservation utility grid
            mu: Logit scale parameter
            function_config: Function configuration
            n_jobs: Number of parallel jobs
            evaluated_pairs: Set of already evaluated pairs (for tracking)
            
        Returns:
            List of evaluation results
        """
        # Prepare arguments for parallel processing
        args_list = []
        for c1 in subset_1:
            for c2 in subset_2:
                # Create unique identifier for this pair
                pair_id = (id(c1), id(c2))
                if pair_id not in evaluated_pairs:
                    args = (c1, c2, V0, mu, self.params, function_config)
                    args_list.append(args)
                    evaluated_pairs.add(pair_id)
        
        if not args_list:
            return []
        
        # Use multiprocessing to process contract pairs in parallel
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.map(DuopolySolver._process_contract_pair_parallel_static, args_list)
        
        return results

    def _merge_pareto_solutions(self, existing_pareto, new_pareto):
        """
        Merge new Pareto solutions with existing ones, removing dominated solutions.
        
        Args:
            existing_pareto: Existing Pareto optimal solutions
            new_pareto: New Pareto optimal solutions
            
        Returns:
            Merged list of Pareto optimal solutions
        """
        if not existing_pareto:
            return new_pareto
        
        if not new_pareto:
            return existing_pareto
        
        # Combine all solutions
        all_solutions = existing_pareto + new_pareto
        
        # Find Pareto optimal subset
        return self.pareto_optimal_solutions(all_solutions)

    def adaptive_grid_search(self, insurer_id, phi1_grid, phi2_grid, 
                           initial_grid_size=10, max_refinements=3, 
                           convergence_threshold=0.05, n_jobs=None):
        """
        Adaptive grid search that starts with coarse grid and refines promising regions.
        
        Args:
            insurer_id: 1 or 2 for the insurer
            phi1_grid: Original phi1 grid
            phi2_grid: Original phi2 grid
            initial_grid_size: Size of initial coarse grid
            max_refinements: Maximum number of refinement iterations
            convergence_threshold: Threshold for convergence
            n_jobs: Number of parallel jobs
            
        Returns:
            List of feasible contracts
        """
        if n_jobs is None:
            n_jobs = mp.cpu_count()
            
        print(f"Starting adaptive grid search for insurer {insurer_id}...")
        
        # Start with coarse grid
        phi1_coarse = np.linspace(phi1_grid[0], phi1_grid[-1], initial_grid_size)
        phi2_coarse = self._create_coarse_phi2_grid(phi2_grid, initial_grid_size)
        
        all_contracts = []
        
        for refinement in range(max_refinements):
            print(f"Refinement {refinement + 1}/{max_refinements}")
            
            # Evaluate current grid
            current_contracts = self.grid_search_contracts_parallel(
                insurer_id, phi1_coarse, phi2_coarse, n_jobs=n_jobs
            )
            
            all_contracts.extend(current_contracts)
            
            if refinement == max_refinements - 1:
                break
            
            # Identify promising regions for refinement
            promising_regions = self._identify_promising_regions(current_contracts)
            
            if not promising_regions:
                print("No promising regions found for refinement")
                break
            
            # Refine grid in promising regions
            phi1_coarse, phi2_coarse = self._refine_grid_in_regions(
                phi1_coarse, phi2_coarse, promising_regions
            )
            
            print(f"Refined grid: {len(phi1_coarse)} phi1 values, {len(phi2_coarse)} phi2 combinations")
        
        return all_contracts

    def _create_coarse_phi2_grid(self, phi2_grid, grid_size):
        """
        Create a coarse phi2 grid for initial evaluation.
        
        Args:
            phi2_grid: Original phi2 grid
            grid_size: Target grid size
            
        Returns:
            Coarse phi2 grid
        """
        if len(phi2_grid) <= grid_size:
            return phi2_grid
        
        # Sample evenly from the original grid
        indices = np.linspace(0, len(phi2_grid) - 1, grid_size, dtype=int)
        return [phi2_grid[i] for i in indices]

    def _identify_promising_regions(self, contracts):
        """
        Identify promising regions based on contract performance.
        
        Args:
            contracts: List of evaluated contracts
            
        Returns:
            List of promising regions (phi1 ranges, phi2 ranges)
        """
        if not contracts:
            return []
        
        # Extract performance metrics
        profits = [c['expected_profit'] for c in contracts]
        phi1_values = [c['phi1'] for c in contracts]
        
        # Find regions with high profits
        profit_threshold = np.percentile(profits, 75)  # Top 25%
        
        promising_phi1 = []
        for i, profit in enumerate(profits):
            if profit >= profit_threshold:
                promising_phi1.append(phi1_values[i])
        
        if not promising_phi1:
            return []
        
        # Group into regions
        regions = []
        phi1_min, phi1_max = min(promising_phi1), max(promising_phi1)
        regions.append({'phi1_range': (phi1_min, phi1_max)})
        
        return regions

    def _refine_grid_in_regions(self, phi1_coarse, phi2_coarse, promising_regions):
        """
        Refine the grid in promising regions.
        
        Args:
            phi1_coarse: Current coarse phi1 grid
            phi2_coarse: Current coarse phi2 grid
            promising_regions: List of promising regions
            
        Returns:
            Tuple of (refined_phi1, refined_phi2)
        """
        refined_phi1 = list(phi1_coarse)
        refined_phi2 = list(phi2_coarse)
        
        for region in promising_regions:
            phi1_min, phi1_max = region['phi1_range']
            
            # Add more points in the promising region
            region_points = 5  # Number of additional points to add
            for i in range(region_points):
                phi1_new = phi1_min + (phi1_max - phi1_min) * (i + 1) / (region_points + 1)
                if phi1_new not in refined_phi1:
                    refined_phi1.append(phi1_new)
        
        # Sort phi1 grid
        refined_phi1.sort()
        
        return refined_phi1, refined_phi2

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

def run_simulation(function_configs=None, include_sensitivity=True, save_plots=True, params=None, logger: SimulationLogger = None, use_parallel=True, n_jobs=None):
    """
    Unified simulation function that can handle single or multiple function configurations.
    
    Args:
        function_configs: List of function configurations to test. 
                         Example: {
                             'name': 'basic_binary_states',
                             'p': 'logistic',
                             'm': 'linear',
                             'e': 'power',
                             'u': 'exponential',
                             'f': 'binary_states',
                             'c': 'linear'
                         }
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
               - u_max: Maximum value parameter for exponential utility function (upper limit)
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
    
    if function_configs is None:
        raise ValueError("function_configs is required and cannot be None")

    results = {}
    # --- Initialize logger if not provided ---
    if logger is None:
        logger = SimulationLogger(experiment_name="duopoly_experiment")
    logger.log_experiment_start("Duopoly insurance simulation")
    logger.log_simulation_settings({
        'function_configs': function_configs,
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
    for config in function_configs:
        print(f"\nRunning simulation with {config['name']} configuration...")
        # Validate that all required function types are specified
        required_keys = ['name', 'p', 'm', 'e', 'u', 'f', 'c']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Function configuration '{config['name']}' is missing required keys: {missing_keys}")
        
        # Create function configuration dictionary (remove 'name' as it's not needed by FlexibleFunctions)
        function_config = {key: config[key] for key in ['p', 'm', 'e', 'u', 'f', 'c']}
        functions = FlexibleFunctions(function_config)
        solver = DuopolySolver(functions, params)
        
        # Choose between parallel and sequential grid search
        if use_parallel:
            print("Using parallel brute-force grid search for contracts...")
            pareto_solutions = solver.brute_force_duopoly(logger=logger, n_jobs=n_jobs)
        else:
            print("Using sequential brute-force grid search for contracts...")
            pareto_solutions = solver.brute_force_duopoly(logger=logger)
            
        results[config['name']] = pareto_solutions
        if save_plots:
            # Only plot the first Pareto solution for illustration
            if pareto_solutions:
                solution = pareto_solutions[0]
                simulator = InsuranceSimulator(solver)
                simulator.plot_results(solution, str(output_dir / f'{config["name"]}_results.png'))
                simulator.plot_discrete_analysis(solution, str(output_dir / f'{config["name"]}_analysis.png'))
        # Log duopoly solution
        if pareto_solutions:
            logger.log_duopoly_solution(pareto_solutions[0], config['name'])
        else:
            logger.log_warning(f"No Pareto solutions found for configuration: {config['name']}")
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
        'u_max': 2*10e3,        # Maximum value parameter for exponential utility function (upper limit)
        'f_p_base': 0.5,        # Parameter for state density function
        'c_lambda': 30,         # Parameter for insurer cost function
        'xi_scale': 500.0,      # Scale parameter for logit choice model
        # Grid search parameters (smaller for faster testing)
        'n_phi1_grid': 10,      # Number of premium grid points
        'n_phi2_grid': 10       # Number of indemnity grid points
    }
    
    # Test action bounds functionality
    print("\nTesting action bounds functionality...")
    function_config = {'p': 'logistic', 'm': 'linear', 'e': 'power', 'u': 'exponential', 'f': 'binary_states', 'c': 'linear'}
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
    
    # Example of multiple function configurations - user must specify all function types
    function_configs = [
        {
            'name': 'basic_binary_states',
            'p': 'logistic',
            'm': 'linear', 
            'e': 'power',
            'u': 'exponential',
            'f': 'binary_states',
            'c': 'linear'
        }
    ]
    
    results = run_simulation(
        function_configs=function_configs,
        include_sensitivity=False,
        params=required_params,
        logger=logger,
        use_parallel=True,
        n_jobs=None  # Use all available cores
    )
    
    print("\n" + "=" * 60)
    print("Simulations completed! Check the generated plots and logs.")
    print("=" * 60)
