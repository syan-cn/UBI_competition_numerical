"""
Duopoly Insurance Model Framework - KKT Solver Implementation

A framework for solving duopoly-style insurance models using KKT conditions
with different functional forms and parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, Tuple, List
from logger import SimulationLogger
from pathlib import Path
import time
import pyomo.environ as pyo

# Pyomo imports for KKT-based solving
try:
    from pyomo.opt import SolverStatus, TerminationCondition
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    print("Warning: Pyomo not available. KKT-based solving will not work.")


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
        
        prob_1 = (1 - delta) * p_base + a * delta
        prob_0 = 1.0 - prob_1
        
        return DiscreteStateSpace([0.0, 1.0], [prob_0, prob_1])

    @staticmethod
    def df_da_binary_states(a: float, delta: float, params: Dict) -> np.ndarray:
        # dP(z=1)/da = delta, dP(z=0)/da = -delta
        return np.array([-delta, delta])


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
        return alpha + beta * a
    
    @staticmethod
    def dp_da_linear(a: float, params: Dict) -> float:
        beta = params['p_beta']
        return beta
    
    @staticmethod
    def d2p_da2_linear(a: float, params: Dict) -> float:
        return 0.0  # Second derivative is zero for linear
    
    @staticmethod
    def exponential(a: float, params: Dict) -> float:
        """Exponential: p(a) = 1 - alpha * exp(-beta * a) (no accident probability increases with action)"""
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for exponential no accident probability function")
        if 'p_beta' not in params:
            raise ValueError("Parameter 'p_beta' is required for exponential no accident probability function")
        
        alpha = params['p_alpha']
        beta = params['p_beta']
        return 1 - alpha * pyo.exp(-beta * a)
    
    @staticmethod
    def dp_da_exponential(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        return alpha * beta * pyo.exp(-beta * a)
    
    @staticmethod
    def d2p_da2_exponential(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        return -alpha * beta**2 * pyo.exp(-beta * a)
    
    @staticmethod
    def power(a: float, params: Dict) -> float:
        """Power: p(a) = 1 - alpha * (1 + a)^(-beta) (no accident probability increases with action)"""
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for power no accident probability function")
        if 'p_beta' not in params:
            raise ValueError("Parameter 'p_beta' is required for power no accident probability function")
        
        alpha = params['p_alpha']
        beta = params['p_beta']
        return 1 - alpha * (1 + a)**(-beta)
    
    @staticmethod
    def dp_da_power(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        return alpha * beta * (1 + a) ** (-(beta + 1))
    
    @staticmethod
    def d2p_da2_power(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        return -alpha * beta * (beta + 1) * (1 + a) ** (-(beta + 2))
    
    @staticmethod
    def logistic(a: float, params: Dict) -> float:
        """Logistic: p(a) = 1 / (1 + exp(-beta * (a - alpha))) (no accident probability increases with action)"""
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for logistic no accident probability function")
        if 'p_beta' not in params:
            raise ValueError("Parameter 'p_beta' is required for logistic no accident probability function")
        
        alpha = params['p_alpha']
        beta = params['p_beta']
        return 1 / (1 + pyo.exp(-beta * (a - alpha)))
    
    @staticmethod
    def dp_da_logistic(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        exp_term = pyo.exp(-beta * (a - alpha))
        denom = (1 + exp_term) ** 2
        return beta * exp_term / denom
    
    @staticmethod
    def d2p_da2_logistic(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        exp_term = pyo.exp(-beta * (a - alpha))
        return -beta**2 * exp_term * (exp_term - 1) / (1 + exp_term)**3


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
        return gamma * pyo.exp(delta)
    
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
    
    @staticmethod
    def d2e_da2_power(a: float, theta: float, params: Dict) -> float:
        kappa = params['e_kappa']
        power = params['e_power']
        if power > 1:
            return kappa * power * (power - 1) * a ** (power - 2) * theta
        else:
            return 0.0


class Utility:
    """Different functional forms for utility u(x)."""
    
    @staticmethod
    def exponential(x: float, params: Dict) -> float:
        """Exponential: u(x) = max * (1 - exp(-rho * x))."""
        if 'u_rho' not in params:
            raise ValueError("Parameter 'u_rho' is required for exponential utility function")
        if 'u_max' not in params:
            raise ValueError("Parameter 'u_max' is required for exponential utility function")
        
        rho = params['u_rho']
        max_val = params['u_max']
        return max_val * (1 - pyo.exp(-rho * x))
    
    @staticmethod
    def du_dx_exponential(x: float, params: Dict) -> float:
        """First derivative of exponential utility: u'(x) = max * rho * exp(-rho * x)"""
        rho = params['u_rho']
        max_val = params['u_max']
        return max_val * rho * pyo.exp(-rho * x)
    
    @staticmethod
    def d2u_dx2_exponential(x: float, params: Dict) -> float:
        """Second derivative of exponential utility: u''(x) = -max * rho^2 * exp(-rho * x)"""
        rho = params['u_rho']
        max_val = params['u_max']
        return -max_val * rho**2 * pyo.exp(-rho * x)
    
    @staticmethod
    def power(x: float, params: Dict) -> float:
        """Power: u(x) = x^rho for rho < 1"""
        if 'u_rho' not in params:
            raise ValueError("Parameter 'u_rho' is required for power utility function")
        
        rho = params['u_rho']
        if x <= 0:
            raise ValueError("x must be positive for power utility function")
        return x**rho
    
    @staticmethod
    def du_dx_power(x: float, params: Dict) -> float:
        """First derivative of power utility: u'(x) = rho * x^(rho-1)"""
        rho = params['u_rho']
        if x <= 0:
            raise ValueError("x must be positive for power utility function")
        return rho * x**(rho - 1)
    
    @staticmethod
    def d2u_dx2_power(x: float, params: Dict) -> float:
        """Second derivative of power utility: u''(x) = rho * (rho-1) * x^(rho-2)"""
        rho = params['u_rho']
        if x <= 0:
            raise ValueError("x must be positive for power utility function")
        if rho > 1:
            return rho * (rho - 1) * x**(rho - 2)
        else:
            return 0.0
    
    @staticmethod
    def logarithmic(x: float, params: Dict) -> float:
        """Logarithmic: u(x) = log(1 + x)"""
        return np.log(1 + x) if x > -1 else -np.inf
    
    @staticmethod
    def du_dx_logarithmic(x: float, params: Dict) -> float:
        """First derivative of logarithmic utility: u'(x) = 1/(1 + x)"""
        return 1 / (1 + x) if x > -1 else 0
    
    @staticmethod
    def d2u_dx2_logarithmic(x: float, params: Dict) -> float:
        """Second derivative of logarithmic utility: u''(x) = -1/(1 + x)^2"""
        return -1 / (1 + x)**2 if x > -1 else 0


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
        return lambda_val * pyo.exp(delta)
    
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
    """Flexible function configurator that allows mixing different functional forms for each function independently."""
    
    def __init__(self, function_config: Dict):
        self.config = function_config
        
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
            'power': ActionCost.power
        }
        
        self.u_functions = {
            'exponential': Utility.exponential,
            'power': Utility.power,
            'logarithmic': Utility.logarithmic
        }
        
        self.f_functions = {
            'binary_states': StateDensity.binary_states
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
        V0, V1, V2 = np.array(V0), np.array(V1), np.array(V2)
        
        # Scale utilities by mu directly
        V0_scaled = V0 / mu
        V1_scaled = V1 / mu
        V2_scaled = V2 / mu
        
        expV = np.exp(np.vstack([V0_scaled, V1_scaled, V2_scaled]))
        denom = expV.sum(axis=0)
        return expV / denom
    
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
    
    def d2p_da2(self, a: float, params: Dict) -> float:
        if 'p' not in self.config:
            raise ValueError("Function type 'p' not specified in configuration")
        func_type = self.config['p']
        return getattr(NoAccidentProbability, f"d2p_da2_{func_type}")(a, params)

    def de_da(self, a: float, theta: float, params: Dict) -> float:
        if 'e' not in self.config:
            raise ValueError("Function type 'e' not specified in configuration")
        func_type = self.config['e']
        return getattr(ActionCost, f"de_da_{func_type}")(a, theta, params)
    
    def d2e_da2(self, a: float, theta: float, params: Dict) -> float:
        if 'e' not in self.config:
            raise ValueError("Function type 'e' not specified in configuration")
        func_type = self.config['e']
        return getattr(ActionCost, f"d2e_da2_{func_type}")(a, theta, params)

    def df_da(self, a: float, delta: float, params: Dict) -> np.ndarray:
        if 'f' not in self.config:
            raise ValueError("Function type 'f' not specified in configuration")
        func_type = self.config['f']
        return getattr(StateDensity, f"df_da_{func_type}")(a, delta, params)
    
    def du_dx(self, x: float, params: Dict) -> float:
        if 'u' not in self.config:
            raise ValueError("Function type 'u' not specified in configuration")
        func_type = self.config['u']
        return getattr(Utility, f"du_dx_{func_type}")(x, params)
    
    def d2u_dx2(self, x: float, params: Dict) -> float:
        if 'u' not in self.config:
            raise ValueError("Function type 'u' not specified in configuration")
        func_type = self.config['u']
        return getattr(Utility, f"d2u_dx2_{func_type}")(x, params)


# ============================================================================
# DUOPOLY SOLVER
# ============================================================================

class DuopolySolver:
    """
    Numerical solver for the duopoly insurance model using KKT conditions.
    """
    
    def __init__(self, functions: FlexibleFunctions, params: Dict):
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
        return 70#-result.fun
    
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
    

    
    # ============================================================================
    # COMPLETE MATHEMATICAL MODEL IMPLEMENTATION
    # ============================================================================
    
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
        
        # Use a more stable approach: scale utilities by mu directly
        scaled_utilities = [u / self.mu for u in utilities]
        
        # Compute choice probabilities using Pyomo exp for Pyomo expressions
        exp_utilities = [pyo.exp(u) for u in scaled_utilities]
        denom = sum(exp_utilities)
        
        P0 = exp_utilities[0] / denom
        P1 = exp_utilities[1] / denom
        P2 = exp_utilities[2] / denom
        
        return P0, P1, P2
    
    def compute_dPi_da(self, theta: float, i: int,
                       a1: float, phi1_1: float, phi2_1: np.ndarray,
                       a2: float, phi1_2: float, phi2_2: np.ndarray) -> float:
        """
        Compute derivative of choice probability P_i with respect to action a_i.
        
        Mathematical formulation:
        ∂P_i/∂a_i = (1/μ) P_i (1-P_i) ∂V_i/∂a_i
        """
        P0, P1, P2 = self.compute_choice_probabilities(theta, a1, phi1_1, phi2_1, a2, phi1_2, phi2_2)
        
        # Compute derivative of V_i with respect to a_i (this is essentially G(θ))
        if i == 1:
            dVi_da = self.compute_G_function(theta, a1, phi1_1, phi2_1, self.delta1)
            Pi = P1
        else:
            dVi_da = self.compute_G_function(theta, a2, phi1_2, phi2_2, self.delta2)
            Pi = P2
        
        # Multinomial logit derivative: ∂P_i/∂a_i = (1/μ) P_i (1-P_i) ∂V_i/∂a_i
        return (1 / self.mu) * Pi * (1 - Pi) * dVi_da
    
    def compute_dVi_da(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute derivative of V_i with respect to action a."""
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        dp_da = self.functions.dp_da(a, self.params)
        de_da = self.functions.de_da(a, theta, self.params)
        
        # Get state space and derivatives
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        # Compute integrals
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
        """
        Compute derivative of V_i with respect to premium φ₁.
        
        Mathematical formulation:
        ∂V_i/∂φ₁ = -p(a) u'(W-φ₁) - [1-p(a)] ∫ u'(W-φ₁+φ₂(z)-s) f(z|a,δ) dz
        """
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        
        # Get state space
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        
        # Marginal utility when no accident occurs: -u'(W-φ₁)
        du_dphi1_no_accident = -self.functions.du_dx(self.W - phi1, self.params)
        
        # Expected marginal utility when accident occurs: -∫ u'(W-φ₁+φ₂(z)-s) f(z|a,δ) dz
        integral_du_dphi1_accident = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            du_dphi1_accident = -self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
            integral_du_dphi1_accident += du_dphi1_accident * z_probs[i]
        
        return p_no_accident * du_dphi1_no_accident + p_accident * integral_du_dphi1_accident
    
    def compute_dVi_dphi2(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float, z_idx: int) -> float:
        """
        Compute derivative of V_i with respect to indemnity φ₂(z).
        
        Mathematical formulation:
        ∂V_i/∂φ₂(z) = [1-p(a)] u'(W-φ₁+φ₂(z)-s) f(z|a,δ)
        """
        p_accident = 1 - self.functions.p(a, self.params)
        
        # Get state space
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        
        # Get φ₂(z) for the specific state
        phi2_val = phi2_values[z_idx]
        
        # Marginal utility at accident state: u'(W-φ₁+φ₂(z)-s)
        du_dphi2 = self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
        
        # State probability
        f_z = z_probs[z_idx]
        
        return p_accident * du_dphi2 * f_z
    
    def compute_dPi_dphi1(self, theta: float, i: int,
                          a1: float, phi1_1: float, phi2_1: np.ndarray,
                          a2: float, phi1_2: float, phi2_2: np.ndarray) -> float:
        """
        Compute derivative of choice probability P_i with respect to premium φ₁^i.
        
        Mathematical formulation:
        ∂P_i/∂φ₁^i = (1/μ) P_i (1-P_i) ∂V_i/∂φ₁^i
        """
        P0, P1, P2 = self.compute_choice_probabilities(theta, a1, phi1_1, phi2_1, a2, phi1_2, phi2_2)
        
        # Compute derivative of V_i with respect to φ₁^i
        if i == 1:
            dVi_dphi1 = self.compute_dVi_dphi1(theta, a1, phi1_1, phi2_1, self.delta1)
            Pi = P1
        else:
            dVi_dphi1 = self.compute_dVi_dphi1(theta, a2, phi1_2, phi2_2, self.delta2)
            Pi = P2
        
        # Multinomial logit derivative: ∂P_i/∂φ₁^i = (1/μ) P_i (1-P_i) ∂V_i/∂φ₁^i
        return (1 / self.mu) * Pi * (1 - Pi) * dVi_dphi1
    
    def compute_dPi_dphi2(self, theta: float, i: int, z_idx: int,
                          a1: float, phi1_1: float, phi2_1: np.ndarray,
                          a2: float, phi1_2: float, phi2_2: np.ndarray) -> float:
        """
        Compute derivative of choice probability P_i with respect to indemnity φ₂^i(z).
        
        Mathematical formulation:
        ∂P_i/∂φ₂^i(z) = (1/μ) P_i (1-P_i) ∂V_i/∂φ₂^i(z)
        """
        # Get current choice probabilities
        P0, P1, P2 = self.compute_choice_probabilities(theta, a1, phi1_1, phi2_1, a2, phi1_2, phi2_2)
        
        # Compute derivative of V_i with respect to φ₂^i(z)
        if i == 1:
            dVi_dphi2 = self.compute_dVi_dphi2(theta, a1, phi1_1, phi2_1, self.delta1, z_idx)
            Pi = P1
        else:
            dVi_dphi2 = self.compute_dVi_dphi2(theta, a2, phi1_2, phi2_2, self.delta2, z_idx)
            Pi = P2
        
        # Multinomial logit derivative: ∂P_i/∂φ₂^i(z) = (1/μ) P_i (1-P_i) ∂V_i/∂φ₂^i(z)
        return (1 / self.mu) * Pi * (1 - Pi) * dVi_dphi2
    
    def compute_G_function(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """
        Compute the G(θ) function from the mathematical model.
        
        Mathematical formulation:
        G(θ) = ∂p(a)/∂a [u(W-φ₁) - ∫ u(W-φ₁+φ₂(z)-s) f(z|a,δ) dz] 
               + (1-p(a)) ∫ u(W-φ₁+φ₂(z)-s) ∂f(z|a,δ)/∂a dz 
               - ∂e(a,θ)/∂a
        """
        # Get derivative values
        dp_da = self.functions.dp_da(a, self.params)
        de_da = self.functions.de_da(a, theta, self.params)
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        
        # Get state space and derivatives
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        # Utility when no accident occurs: u(W-φ₁)
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        
        # First integral: ∫ u(W-φ₁+φ₂(z)-s) f(z|a,δ) dz
        integral_1 = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            integral_1 += u_accident * z_probs[i]
        
        # Second integral: ∫ u(W-φ₁+φ₂(z)-s) ∂f(z|a,δ)/∂a dz
        integral_2 = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            integral_2 += u_accident * df_da[i]
        
        g_value = dp_da * (utility_no_accident - integral_1) + p_accident * integral_2 - de_da
        
        return g_value
    
    def compute_dG_dphi1(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """
        Compute derivative of G(θ) with respect to φ₁.
        
        Mathematical formulation:
        ∂G/∂φ₁ = - ∂p/∂a [u'(W-φ₁) - ∫ u'(W-φ₁+φ₂(z)-s) f(z|a,δ) dz] 
                  - (1-p) ∫ u'(W-φ₁+φ₂(z)-s) ∂f(z|a,δ)/∂a dz
        """
        dp_da = self.functions.dp_da(a, self.params)
        p_accident = 1 - self.functions.p(a, self.params)
        
        # Get state space and derivatives
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        # Marginal utility derivatives
        du_no_accident = self.functions.du_dx(self.W - phi1, self.params)  # u'(W-φ₁)
        
        # First integral: ∫ u'(W-φ₁+φ₂(z)-s) f(z|a,δ) dz
        integral_marginal_1 = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            du_accident = self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)  # u'(W-φ₁+φ₂-s)
            integral_marginal_1 += du_accident * z_probs[i]
        
        # Second integral: ∫ u'(W-φ₁+φ₂(z)-s) ∂f(z|a,δ)/∂a dz
        integral_marginal_2 = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            du_accident = self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)  # u'(W-φ₁+φ₂-s)
            integral_marginal_2 += du_accident * df_da[i]
        
        dG_dphi1 = -dp_da * (du_no_accident - integral_marginal_1) - p_accident * integral_marginal_2
        
        return dG_dphi1
    
    def compute_dG_dphi2(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float, z_idx: int) -> float:
        """
        Compute derivative of G(θ) with respect to φ₂(z).
        
        Mathematical formulation:
        ∂G/∂φ₂(z) = - ∂p/∂a [u'(W-φ₁+φ₂(z)-s) f(z|a,δ)] 
                     + (1-p) u'(W-φ₁+φ₂(z)-s) ∂f(z|a,δ)/∂a
        """
        dp_da = self.functions.dp_da(a, self.params)
        p_accident = 1 - self.functions.p(a, self.params)
        
        # Get state space and derivatives
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        # Compute derivative for specific state z_idx
        phi2_val = phi2_values[z_idx]
        
        # Marginal utility at accident state: u'(W-φ₁+φ₂(z)-s)
        du_accident = self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
        
        # State probability and its derivative
        f_z = z_probs[z_idx]
        df_da_z = df_da[z_idx]
        
        dG_dphi2 = - dp_da * (du_accident * f_z) + p_accident * du_accident * df_da_z
        
        return dG_dphi2
    
    def compute_dG_da(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """
        Compute derivative of G(θ) with respect to action a (second-order derivative).
        
        This is the most complex derivative needed for the KKT stationarity conditions.
        
        Mathematical formulation (expanded from G function):
        ∂G/∂a = ∂²p/∂a² [u(W-φ₁) - ∫ u(W-φ₁+φ₂(z)-s) f(z|a,δ) dz]
                - 2 * ∂p/∂a [∫ u(W-φ₁+φ₂(z)-s) ∂f(z|a,δ)/∂a dz]
                + (1-p) ∫ u(W-φ₁+φ₂(z)-s) ∂²f(z|a,δ)/∂a² dz
                - ∂²e/∂a²
        """
        # Get all necessary derivatives
        dp_da = self.functions.dp_da(a, self.params)
        d2p_da2 = self.functions.d2p_da2(a, self.params)
        d2e_da2 = self.functions.d2e_da2(a, theta, self.params)
        
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        
        # Get state space and derivatives
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)
        
        # Note: For discrete states, ∂²f/∂a² would require additional implementation
        # For discrete binary states, second derivatives are mathematically zero
        d2f_da2 = np.zeros_like(df_da)  # Exact for discrete linear state transitions
        
        # Utility when no accident occurs
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        
        # Integral terms
        integral_u_f = 0.0          # ∫ u(W-φ₁+φ₂(z)-s) f(z|a,δ) dz
        integral_u_df_da = 0.0      # ∫ u(W-φ₁+φ₂(z)-s) ∂f(z|a,δ)/∂a dz
        integral_u_d2f_da2 = 0.0    # ∫ u(W-φ₁+φ₂(z)-s) ∂²f(z|a,δ)/∂a² dz
        
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            
            integral_u_f += u_accident * z_probs[i]
            integral_u_df_da += u_accident * df_da[i]
            integral_u_d2f_da2 += u_accident * d2f_da2[i]
        
        # Term 1: ∂²p/∂a² [u(W-φ₁) - ∫ u(W-φ₁+φ₂(z)-s) f(z|a,δ) dz]
        term1 = d2p_da2 * (utility_no_accident - integral_u_f)
        
        # Term 2: -2 * ∂p/∂a [∫ u(W-φ₁+φ₂(z)-s) ∂f(z|a,δ)/∂a dz]
        term2 = -2 * dp_da * integral_u_df_da
        
        # Term 3: (1-p) ∫ u(W-φ₁+φ₂(z)-s) ∂²f(z|a,δ)/∂a² dz
        term3 = p_accident * integral_u_d2f_da2
        
        # Term 4: -∂²e/∂a²
        term4 = -d2e_da2
        
        # Complete mathematical formulation
        dG_da = term1 + term2 + term3 + term4
        
        return dG_da

    def build_and_solve_model(self, solver_name='ipopt', verbose=False, debug_mode=True):
        """Solve the duopoly equilibrium using KKT conditions with pyomo."""
        if not PYOMO_AVAILABLE:
            raise ImportError("Pyomo is required for KKT-based solving. Please install it with: pip install pyomo")
        
        print("Setting up KKT system for duopoly equilibrium...")
        
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
        model.theta_vals = pyo.Param(model.THETA, initialize={i: self.theta_grid[i] for i in range(self.n_theta)})
        model.h_theta = pyo.Param(model.THETA, initialize={i: self.h_theta[i] for i in range(self.n_theta)})
        model.z_vals = pyo.Param(model.Z, initialize={i: self.z_values[i] for i in range(self.n_states)})
        model.delta1 = pyo.Param(initialize=self.delta1)
        model.delta2 = pyo.Param(initialize=self.delta2)
        
        # Decision Variables for both insurers
        model.a = pyo.Var(model.I, model.THETA, domain=pyo.NonNegativeReals, 
                         bounds=(self.a_min, self.a_max))
        model.phi1 = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0.1, self.s))
        model.phi2 = pyo.Var(model.I, model.Z, domain=pyo.NonNegativeReals, bounds=(0.0, self.s))
        
        # Lagrange multipliers
        model.lam = pyo.Var(model.I, model.THETA, domain=pyo.Reals)
        model.nu_L = pyo.Var(model.I, model.THETA, domain=pyo.NonNegativeReals)
        model.nu_U = pyo.Var(model.I, model.THETA, domain=pyo.NonNegativeReals)
        model.eta = pyo.Var(model.I, domain=pyo.NonNegativeReals)
        model.gamma = pyo.Var(model.I, model.Z, domain=pyo.NonNegativeReals)
        
        def incentive_constraint_rule(model, i, t):
            """
            Incentive compatibility constraint G(θ) = 0.
            
            Mathematical formulation:
            G(θ) = ∂p(a)/∂a [u(W-φ₁) - ∫ u(W-φ₁+φ₂(z)-s) f(z|a,δ) dz] 
                   + (1-p(a)) ∫ u(W-φ₁+φ₂(z)-s) ∂f(z|a,δ)/∂a dz 
                   - ∂e(a,θ)/∂a = 0
            """
            theta = model.theta_vals[t]
            a_val = model.a[i, t]
            phi1_val = model.phi1[i]
            delta_val = model.delta1 if i == 1 else model.delta2
            
            # Get phi2 values for all states
            phi2_values = np.array([model.phi2[i, z] for z in model.Z])
            
            # Use the existing compute_G_function method
            G_val = self.compute_G_function(theta, a_val, phi1_val, phi2_values, delta_val)
            
            return G_val == 0
        
        model.incentive_constraint = pyo.Constraint(model.I, model.THETA, rule=incentive_constraint_rule)
        
        # 2. Complementary slackness constraints - RELAXED for numerical feasibility
        def comp_slack_lower_rule(model, i, t):
            # Soft constraint: |nu_L * (a_min - a)| <= tolerance
            return model.nu_L[i, t] * (self.a_min - model.a[i, t]) == 0
        
        model.comp_slack_lower = pyo.Constraint(model.I, model.THETA, rule=comp_slack_lower_rule)
        
        def comp_slack_upper_rule(model, i, t):
            # Soft constraint: |nu_U * (a - a_max)| <= tolerance  
            return model.nu_U[i, t] * (model.a[i, t] - self.a_max) == 0
        
        model.comp_slack_upper = pyo.Constraint(model.I, model.THETA, rule=comp_slack_upper_rule)
        
        def comp_slack_premium_rule(model, i):
            # Soft constraint: |eta * phi1| <= tolerance
            return model.eta[i] * model.phi1[i] == 0
        
        model.comp_slack_premium = pyo.Constraint(model.I, rule=comp_slack_premium_rule)
        
        def comp_slack_indemnity_rule(model, i, z):
            # Already uses soft constraint - keep as is
            return model.gamma[i, z] * model.phi2[i, z] == 0
        
        model.comp_slack_indemnity = pyo.Constraint(model.I, model.Z, rule=comp_slack_indemnity_rule)
        
        # 3. Stationarity conditions
        def stationarity_action_rule(model, i, t):
            """
            Stationarity with respect to action a^i(θ).
            
            Mathematical formulation:
            0 = N ∂P_i(θ)/∂a^i(θ) [profit_term] 
                - N P_i(θ) [second_term_bracket]
                + λ(θ) ∂G(θ)/∂a^i(θ) - ν_L(θ) + ν_U(θ)
            """
            theta = model.theta_vals[t]
            a_val = model.a[i, t]
            phi1_val = model.phi1[i]
            
            # Get monitoring level
            delta_val = model.delta1 if i == 1 else model.delta2
            
            # Get phi2 values for all states
            phi2_values = np.array([model.phi2[i, z] for z in model.Z])
            
            # Get contract values for the other insurer
            other_insurer = 2 if i == 1 else 1
            a_other = model.a[other_insurer, t]
            phi1_other = model.phi1[other_insurer]
            phi2_other_values = np.array([model.phi2[other_insurer, z] for z in model.Z])
            
            # Use existing helper functions to compute choice probabilities
            P0, P1, P2 = self.compute_choice_probabilities(theta, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
            Pi = P1 if i == 1 else P2
            
            # Use existing helper function to compute ∂P_i(θ)/∂a^i(θ)
            dPi_da = self.compute_dPi_da(theta, i, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
            
            # Use existing helper function to compute ∂G(θ)/∂a^i(θ)
            dG_da = self.compute_dG_da(theta, a_val, phi1_val, phi2_values, delta_val)
            
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
            
            # Compute second term bracket: -p'(a^i(θ)) ∫ φ₂^i(z) f(z|a^i(θ),δ^i) dz + (1-p(a^i(θ))) ∫ φ₂^i(z) ∂f(z|a^i(θ),δ^i)/∂a^i(θ) dz
            dp_da = self.functions.dp_da(a_val, self.params)
            df_da = self.functions.df_da(a_val, delta_val, self.params)
            
            integral_phi2_df_da = 0.0
            for j, z in enumerate(z_values):
                integral_phi2_df_da += phi2_values[j] * df_da[j]
            
            second_term_bracket = -dp_da * integral_phi2_f + p_accident * integral_phi2_df_da
            
            # KKT stationarity condition according to mathematical model
            term1 = self.N * dPi_da * profit_term
            term2 = -self.N * Pi * second_term_bracket
            term3 = model.lam[i, t] * dG_da
            
            return term1 + term2 + term3 - model.nu_L[i, t] + model.nu_U[i, t] == 0
        
        model.stationarity_action = pyo.Constraint(model.I, model.THETA, rule=stationarity_action_rule)
        
        def stationarity_premium_rule(model, i):
            """
            Stationarity with respect to premium φ₁^i.
            
            Mathematical formulation:
            0 = N ∫ h(θ) [∂P_i(θ)/∂φ₁^i * profit_term + P_i(θ)] dθ + ∫ λ(θ) ∂G(θ)/∂φ₁^i h(θ) dθ - η^i
            """
            # Get monitoring level for this insurer
            delta_val = model.delta1 if i == 1 else model.delta2
            
            # Calculate complete integral over all risk types
            integral_term = 0.0
            lambda_dG_integral = 0.0
            
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
                
                # Use existing helper functions
                P0, P1, P2 = self.compute_choice_probabilities(theta, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
                Pi = P1 if i == 1 else P2
                
                dPi_dphi1 = self.compute_dPi_dphi1(theta, i, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
                dG_dphi1 = self.compute_dG_dphi1(theta, a_val, phi1_val, phi2_values, delta_val)
                
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
                
                # First integral term: ∫ h(θ) [∂P_i(θ)/∂φ₁^i * profit_term + P_i(θ)] dθ
                integral_term += h_theta_val * (dPi_dphi1 * profit_term + Pi)
                
                # Second integral term: ∫ λ(θ) ∂G(θ)/∂φ₁^i h(θ) dθ
                lambda_dG_integral += model.lam[i, t] * dG_dphi1 * h_theta_val
            
            # Stationarity condition
            return self.N * integral_term + lambda_dG_integral - model.eta[i] == 0
        
        model.stationarity_premium = pyo.Constraint(model.I, rule=stationarity_premium_rule)
        
        def stationarity_indemnity_rule(model, i, z):
            """
            Stationarity with respect to indemnity φ₂^i(z).
            
            Mathematical formulation:
            0 = -N ∫ h(θ) P_i(θ) (1-p(a^i(θ))) f(z|a^i(θ),δ^i) dθ 
                + N ∫ h(θ) ∂P_i(θ)/∂φ₂^i(z) [profit_term] dθ 
                + ∫ λ(θ) ∂G(θ)/∂φ₂^i(z) h(θ) dθ - γ^i(z)
            """
            # Get monitoring level for this insurer
            delta_val = model.delta1 if i == 1 else model.delta2
            
            # Calculate complete integrals over all risk types
            first_integral = 0.0     # ∫ h(θ) P_i(θ) (1-p(a^i(θ))) f(z|a^i(θ),δ^i) dθ
            second_integral = 0.0    # ∫ h(θ) ∂P_i(θ)/∂φ₂^i(z) [profit_term] dθ
            lambda_dG_integral = 0.0 # ∫ λ(θ) ∂G(θ)/∂φ₂^i(z) h(θ) dθ
            
            for t in model.THETA:
                theta = model.theta_vals[t]
                h_theta_val = model.h_theta[t]
                a_val = model.a[i, t]
                phi1_val = model.phi1[i]
                
                # Get phi2 values for all states
                phi2_values = np.array([model.phi2[i, z_idx] for z_idx in model.Z])
                
                # Get contract values for the other insurer
                other_insurer = 2 if i == 1 else 1
                a_other = model.a[other_insurer, t]
                phi1_other = model.phi1[other_insurer]
                phi2_other_values = np.array([model.phi2[other_insurer, z_idx] for z_idx in model.Z])
                
                # Use existing helper functions
                P0, P1, P2 = self.compute_choice_probabilities(theta, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
                Pi = P1 if i == 1 else P2
                
                dPi_dphi2 = self.compute_dPi_dphi2(theta, i, z, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
                dG_dphi2 = self.compute_dG_dphi2(theta, a_val, phi1_val, phi2_values, delta_val, z)
                
                # Compute profit term: φ₁^i - (1-p(a^i(θ))) ∫ φ₂^i(t) f(t|a^i(θ),δ^i) dt
                p_no_accident = self.functions.p(a_val, self.params)
                p_accident = 1 - p_no_accident
                
                # Get state space and compute integral
                state_space = self.functions.f(a_val, delta_val, self.params)
                z_values, z_probs = state_space.get_all_states()
                
                integral_phi2_f = 0.0
                for j, z_val in enumerate(z_values):
                    integral_phi2_f += phi2_values[j] * z_probs[j]
                
                profit_term = phi1_val - p_accident * integral_phi2_f
                
                # Get f(z|a^i(θ),δ^i) for the specific state z
                f_z_val = z_probs[z]
                
                # First integral: ∫ h(θ) P_i(θ) (1-p(a^i(θ))) f(z|a^i(θ),δ^i) dθ
                first_integral += h_theta_val * Pi * p_accident * f_z_val
                
                # Second integral: ∫ h(θ) ∂P_i(θ)/∂φ₂^i(z) [profit_term] dθ
                second_integral += h_theta_val * dPi_dphi2 * profit_term
                
                # Third integral: ∫ λ(θ) ∂G(θ)/∂φ₂^i(z) h(θ) dθ
                lambda_dG_integral += model.lam[i, t] * dG_dphi2 * h_theta_val
            
            # Stationarity condition
            return -self.N * first_integral + self.N * second_integral + lambda_dG_integral - model.gamma[i, z] == 0
        
        model.stationarity_indemnity = pyo.Constraint(model.I, model.Z, rule=stationarity_indemnity_rule)
        
        # Objective function - 0
        
        model.obj = pyo.Objective(expr=0, sense=pyo.minimize)
        
        # Initial values
        for i in model.I:
            model.phi1[i].set_value(self.s * 0.3)  # Start with lower premium
            for z in model.Z:
                model.phi2[i, z].set_value(self.s * 0.2)  # Start with lower indemnity
            for t in model.THETA:
                model.a[i, t].set_value(0.5)  # Start with middle action
                model.lam[i, t].set_value(0.1)  # Small positive multiplier
                model.nu_L[i, t].set_value(0.0)  # Start with zero
                model.nu_U[i, t].set_value(0.0)  # Start with zero
            model.eta[i].set_value(0.1)  # Small positive multiplier
            for z in model.Z:
                model.gamma[i, z].set_value(0.1)  # Small positive multiplier
        
        # Debug mode: Analyze model structure
        if debug_mode:
            print("\n" + "="*50)
            print("DEBUG MODE: MODEL ANALYSIS")
            print("="*50)
            
            # Count variables and constraints
            total_vars = len(model.component_map(pyo.Var))
            total_constrs = len(model.component_map(pyo.Constraint))
            
            print(f"Model Statistics:")
            print(f"  - Variables: {total_vars}")
            print(f"  - Constraints: {total_constrs}")
            print(f"  - C/V Ratio: {total_constrs/total_vars:.2f}")
            
            # Check variable bounds
            print(f"\nVariable Bounds Check:")
            for var_name, var_obj in model.component_map(pyo.Var).items():
                print(f"  {var_name}:")
                for idx in var_obj.index_set():
                    var = var_obj[idx]
                    lb = var.lb
                    ub = var.ub
                    if lb is not None and ub is not None and lb > ub:
                        print(f"    WARNING: Bound conflict at {var_name}[{idx}]: lb={lb}, ub={ub}")
                    elif lb is not None and ub is not None and (ub - lb) < 1e-6:
                        print(f"    WARNING: Tight bounds at {var_name}[{idx}]: range={ub-lb:.2e}")
        
        # Solve the model
        print(f"\nSolving KKT system with {solver_name} solver...")
        
        try:
            opt = pyo.SolverFactory(solver_name)
            if solver_name == 'ipopt':
                opt.options.update({
                    'max_iter': 100000 if debug_mode else 50000,  # More iterations for complex problems
                    'tol': 1e-6 if debug_mode else 1e-5,  # Relaxed tolerance
                    'print_level': 5 if verbose or debug_mode else 0,
                    'output_file': 'ipopt_debug.out' if debug_mode else None,
                    'linear_solver': 'mumps',  # Good for economic problems
                    'hessian_approximation': 'limited-memory',
                    'mu_strategy': 'adaptive',
                    'bound_push': 1e-6,  # Relaxed for better convergence
                    'bound_frac': 1e-6,
                    'slack_bound_push': 1e-6,
                    'slack_bound_frac': 1e-6,
                    'dual_inf_tol': 1e-6,  # Relaxed dual infeasibility tolerance
                    'compl_inf_tol': 1e-6,  # Relaxed complementarity tolerance
                    'acceptable_tol': 1e-5,  # Relaxed acceptable tolerance
                    'acceptable_iter': 15,  # More acceptable iterations
                    'alpha_for_y': 'primal',  # Better for economic problems
                    'recalc_y': 'yes',  # Recalculate dual variables
                    'mehrotra_algorithm': 'yes',  # Use Mehrotra's algorithm
                    'warm_start_init_point': 'yes',  # Use warm start
                    'warm_start_bound_push': 1e-6,
                    'warm_start_bound_frac': 1e-6,
                    'warm_start_slack_bound_push': 1e-6,
                    'warm_start_slack_bound_frac': 1e-6
                })
            
            import time
            start_time = time.time()
            
            results = opt.solve(model, tee=verbose or debug_mode)
            
            solve_time = time.time() - start_time
            
            # Enhanced result analysis
            if debug_mode:
                print(f"\n" + "="*50)
                print("DEBUG MODE: SOLVER RESULTS ANALYSIS")
                print("="*50)
                print(f"Solver Status: {results.solver.status}")
                print(f"Termination Condition: {results.solver.termination_condition}")
                print(f"Solve Time: {solve_time:.3f} seconds")
                
                if hasattr(results.solver, 'iterations'):
                    print(f"Iterations: {results.solver.iterations}")
                
                if hasattr(results.solver, 'objective_value'):
                    print(f"Objective Value: {results.solver.objective_value}")
                
                # Check constraint violations
                print(f"\nConstraint Violation Analysis:")
                max_violation = 0.0
                num_violated = 0
                
                for constr_name, constr_obj in model.component_map(pyo.Constraint).items():
                    for idx in constr_obj.index_set():
                        constr = constr_obj[idx]
                        try:
                            body_value = pyo.value(constr.body)
                            lower_value = pyo.value(constr.lower) if constr.lower is not None else None
                            upper_value = pyo.value(constr.upper) if constr.upper is not None else None
                            
                            violation = 0.0
                            if lower_value is not None and body_value < lower_value:
                                violation = lower_value - body_value
                            elif upper_value is not None and body_value > upper_value:
                                violation = body_value - upper_value
                            
                            if violation > 1e-6:
                                num_violated += 1
                                max_violation = max(max_violation, violation)
                                if num_violated <= 5:  # Show first 5 violations
                                    print(f"  {constr_name}[{idx}]: violation={violation:.2e}")
                                    
                        except Exception as e:
                            print(f"  Error evaluating {constr_name}[{idx}]: {e}")
                
                print(f"Total violations: {num_violated}")
                print(f"Maximum violation: {max_violation:.2e}")
                
                # Read detailed solver output if available
                try:
                    with open('ipopt_debug.out', 'r') as f:
                        solver_output = f.read()
                        print(f"\nDetailed Solver Output (last 20 lines):")
                        lines = solver_output.strip().split('\n')
                        for line in lines[-20:]:
                            print(f"  {line}")
                except FileNotFoundError:
                    print("No detailed solver output file found.")
            
            if (results.solver.status == SolverStatus.ok and 
                results.solver.termination_condition == TerminationCondition.optimal):
                
                print("KKT system solved successfully!")
                
                # Extract solution
                solution = {
                    'insurer1': {
                        'phi1': pyo.value(model.phi1[1]),
                        'phi2': np.array([pyo.value(model.phi2[1, z]) for z in model.Z]),
                        'a_schedule': np.array([pyo.value(model.a[1, t]) for t in model.THETA]),
                        'multipliers': {
                            'lambda': np.array([pyo.value(model.lam[1, t]) for t in model.THETA]),
                            'nu_L': np.array([pyo.value(model.nu_L[1, t]) for t in model.THETA]),
                            'nu_U': np.array([pyo.value(model.nu_U[1, t]) for t in model.THETA]),
                            'eta': pyo.value(model.eta[1]),
                            'gamma': np.array([pyo.value(model.gamma[1, z]) for z in model.Z])
                        }
                    },
                    'insurer2': {
                        'phi1': pyo.value(model.phi1[2]),
                        'phi2': np.array([pyo.value(model.phi2[2, z]) for z in model.Z]),
                        'a_schedule': np.array([pyo.value(model.a[2, t]) for t in model.THETA]),
                        'multipliers': {
                            'lambda': np.array([pyo.value(model.lam[2, t]) for t in model.THETA]),
                            'nu_L': np.array([pyo.value(model.nu_L[2, t]) for t in model.THETA]),
                            'nu_U': np.array([pyo.value(model.nu_U[2, t]) for t in model.THETA]),
                            'eta': pyo.value(model.eta[2]),
                            'gamma': np.array([pyo.value(model.gamma[2, z]) for z in model.Z])
                        }
                    },
                    'solver_status': 'optimal',
                    'solver_info': results,
                    'solve_time': solve_time
                }
                
                return solution
                
            else:
                print(f"\nSolver failed!")
                print(f"  Status: {results.solver.status}")
                print(f"  Termination: {results.solver.termination_condition}")
                
                # Provide specific guidance based on termination condition
                if results.solver.termination_condition == TerminationCondition.infeasible:
                    print(f"  REASON: Problem is infeasible")
                elif results.solver.termination_condition == TerminationCondition.infeasibleOrUnbounded:
                    print(f"  REASON: Problem is infeasible or unbounded")
                elif results.solver.termination_condition == TerminationCondition.maxIterations:
                    print(f"  REASON: Maximum iterations reached")
                elif results.solver.termination_condition == TerminationCondition.other:
                    print(f"  REASON: Other termination condition")
                
                return None
                
        except Exception as e:
            print(f"Error solving KKT system: {e}")
            if debug_mode:
                import traceback
                print("Full traceback:")
                traceback.print_exc()
            return None

    def run(self, solver_name='ipopt', verbose=True, save_plots=True, logger=None):
        """
        Run KKT-based simulation for duopoly insurance model.
        
        Args:
            solver_name: Name of the solver to use
            verbose: Whether to print detailed output
            save_plots: Whether to save plots to files
            logger: SimulationLogger instance for recording results
            
        Returns:
            Tuple of (success, solution) where solution is a dictionary containing simulation results
        """
        print("Testing KKT solver...")
        print(f"Model configuration:")
        print(f"  - Risk types: {self.n_theta}")
        print(f"  - States: {self.n_states}")
        print(f"  - Action bounds: [{self.a_min}, {self.a_max}]")
        print(f"  - Initial wealth W: {self.W}")
        print(f"  - Accident severity s: {self.s}")
        print(f"  - Monitoring levels: δ₁={self.delta1}, δ₂={self.delta2}")

        # Log experiment start if logger is provided
        if logger:
            logger.log_experiment_start("Duopoly insurance simulation")
            logger.log_simulation_settings({
                'solver_name': solver_name,
                'save_plots': save_plots,
                'verbose': verbose
            })
            logger.log_parameters(self.params)

        try:
            start_time = time.time()
            solution = self.build_and_solve_model(solver_name=solver_name, verbose=verbose)
            solve_time = time.time() - start_time

            if solution is not None:
                print("\n" + "="*50)
                print("KKT SOLVER SUCCESS!")
                print("="*50)

                for insurer_id in [1, 2]:
                    key = f'insurer{insurer_id}'
                    print(f"\nInsurer {insurer_id} Solution:")
                    print(f"  Premium φ₁^{insurer_id}: {solution[key]['phi1']:.4f}")
                    print(f"  Indemnities φ₂^{insurer_id}: {solution[key]['phi2']}")
                    print(f"  Action schedule a^{insurer_id}(θ): {solution[key]['a_schedule']}")
                    print(f"  Average action: {np.mean(solution[key]['a_schedule']):.4f}")

                # Log performance metrics if logger is provided
                if logger:
                    logger.log_performance_metric("solve_time_seconds", solve_time)
                    
                    # Log the solution
                    solution_for_logging = {
                        'insurer1': {
                            'premium': solution['insurer1']['phi1'],
                            'indemnity_values': solution['insurer1']['phi2'],
                            'a_schedule': solution['insurer1']['a_schedule'],
                            'expected_profit': 0  # Could compute this if needed
                        },
                        'insurer2': {
                            'premium': solution['insurer2']['phi1'],
                            'indemnity_values': solution['insurer2']['phi2'],
                            'a_schedule': solution['insurer2']['a_schedule'],
                            'expected_profit': 0  # Could compute this if needed
                        },
                        'total_profit': 0,
                        'market_share': {}
                    }
                    logger.log_duopoly_solution(solution_for_logging, "single_configuration")

                # Generate plots if requested
                if save_plots:
                    plots_dir = Path("outputs") / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Comprehensive results plot
                    plot_path = plots_dir / "comprehensive_results.png"
                    plot_results(self, solution, save_path=str(plot_path))
                    
                    if logger:
                        logger.log_plot_generation("comprehensive_results", str(plot_path), "single_configuration")

                # Log experiment end if logger is provided
                if logger:
                    logger.log_experiment_end("Simulation completed successfully")
                    logger.print_summary()

                return True, solution
            else:
                print("KKT solver failed to find a solution.")
                
                if logger:
                    logger.log_error("Solver failed to converge")
                    logger.log_experiment_end("Simulation failed")
                
                return False, None

        except Exception as e:
            error_msg = f"Error testing KKT solver: {e}"
            print(error_msg)
            
            if logger:
                logger.log_error(error_msg)
                logger.log_experiment_end("Simulation failed with error")
            
            return False, None


# ============================================================================
# SIMULATION & PLOTTING
# ============================================================================

def plot_results(solver: DuopolySolver, solution: Dict, save_path: str = None):
    """Comprehensive plotting of simulation results with multiple visualization types."""
    if solution is None:
        print("No solution to plot.")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Action schedules comparison
    theta_grid = solver.theta_grid
    a1 = solution['insurer1']['a_schedule']
    a2 = solution['insurer2']['a_schedule']
    
    axes[0, 0].plot(theta_grid, a1, 'b-', linewidth=2, label='Insurer 1', marker='o')
    axes[0, 0].plot(theta_grid, a2, 'r--', linewidth=2, label='Insurer 2', marker='s')
    axes[0, 0].set_xlabel(r'$\theta$ (Risk Type)')
    axes[0, 0].set_ylabel(r'$a^i(\theta)$ (Action Level)')
    axes[0, 0].set_title('Optimal Action Schedules')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Premium comparison
    phi1_1 = solution['insurer1']['phi1']
    phi1_2 = solution['insurer2']['phi1']
    
    bars = axes[0, 1].bar(['Insurer 1', 'Insurer 2'], [phi1_1, phi1_2], 
                         color=['blue', 'red'], alpha=0.7)
    axes[0, 1].set_ylabel(r'$\phi_1^i$ (Premium)')
    axes[0, 1].set_title('Optimal Premiums')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 3: Indemnity schedules
    z_values = solver.z_values
    phi2_1 = solution['insurer1']['phi2']
    phi2_2 = solution['insurer2']['phi2']
    
    x_pos = np.arange(len(z_values))
    width = 0.35
    
    # Indemnity schedules now occupy the first subplot of the second row
    axes[1, 0].bar(x_pos - width/2, phi2_1, width, label='Insurer 1', alpha=0.7, color='blue')
    axes[1, 0].bar(x_pos + width/2, phi2_2, width, label='Insurer 2', alpha=0.7, color='red')
    axes[1, 0].set_xlabel('State Index')
    axes[1, 0].set_ylabel(r'$\phi_2^i(z)$ (Indemnity)')
    axes[1, 0].set_title('Discrete Indemnity Schedules')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f'z={z:.1f}' for z in z_values])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Expected utilities for different risk types
    V0_list = []
    V1_list = []
    V2_list = []
    for theta in theta_grid:
        # Compute expected utilities for each risk type using complete utility calculation
        a1_theta = np.interp(theta, theta_grid, a1)
        a2_theta = np.interp(theta, theta_grid, a2)
        
        V0 = solver.compute_reservation_utility(theta)
        V1 = solver.compute_expected_utility(a1_theta, phi1_1, phi2_1, solver.delta1, theta)
        V2 = solver.compute_expected_utility(a2_theta, phi1_2, phi2_2, solver.delta2, theta)
        V0_list.append(V0)
        V1_list.append(V1)
        V2_list.append(V2)
    
    axes[1, 1].plot(theta_grid, V0_list, 'k-', linewidth=2, label='No Insurance', marker='^')
    axes[1, 1].plot(theta_grid, V1_list, 'b-', linewidth=2, label='Insurer 1', marker='o')
    axes[1, 1].plot(theta_grid, V2_list, 'r--', linewidth=2, label='Insurer 2', marker='s')
    axes[1, 1].set_xlabel(r'$\theta$ (Risk Type)')
    axes[1, 1].set_ylabel('Utility')
    axes[1, 1].set_title('Market Coverage Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example parameters
    params = {
        'W': 100.0,            # Initial wealth (reduced from 1000.0)
        's': 30.0,             # Accident severity (reduced from 300.0)
        'N': 100,              # Number of customers
        'delta1': 0.9,         # Insurer 1 monitoring level
        'delta2': 0.2,         # Insurer 2 monitoring level
        'theta_min': 0.05,      # Minimum risk type
        'theta_max': 1,      # Maximum risk type
        'n_theta': 10,          # Number of risk types
        'a_min': 0.0,          # Minimum action level
        'a_max': 1.0,          # Maximum action level
        'mu': 100.0,             # Logit model scale parameter
        'p_alpha': 0.0,        # No accident probability parameter
        'p_beta': 1.0,         # No accident probability parameter
        'e_kappa': 30.0,       # Action cost parameter (reduced from 200.0)
        'e_power': 2.0,        # Action cost power
        'f_p_base': 0.5,       # State density parameter
        'c_lambda': 1.0,       # Insurer cost parameter (reduced from 10.0)
        'm_gamma': 2.0,        # Monitoring cost parameter (reduced from 20.0)
        'u_rho': 0.05,          # Utility parameter (increased from 0.05 for better scaling)
        'u_max': 100.0,        # Utility parameter (reduced from 1000.0)
    }
    
    print("="*60)
    print("DUOPOLY INSURANCE MODEL DEMONSTRATION")
    print("="*60)

    # Create logger for analysis
    from logger import SimulationLogger
    logger = SimulationLogger(
        experiment_name="duopoly_kkt_demo",
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
    
    # Create solver with the function configuration
    functions = FlexibleFunctions(function_config)
    solver = DuopolySolver(functions, params)
    
    # Run simulation using the simplified run method
    success, solution = solver.run(
        solver_name='ipopt',
        verbose=False,
        save_plots=True,
        logger=logger
    )
    
    if success:
        print("✅ KKT-based simulation completed!")
        print(f"Solve time: {solution.get('solve_time', 'N/A')} seconds")
    else:
        print("❌ KKT-based simulation failed!")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED")
    print("="*60)
