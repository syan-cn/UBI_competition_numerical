"""
Discrete State Space Module

This module contains classes and functions for handling discrete state spaces
in insurance models, including state density functions.
"""

import numpy as np
import pyomo.environ as pyo
from typing import Dict, Tuple, List
from scipy.special import comb


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
    
    @staticmethod
    def binomial_states(a: float, delta: float, params: Dict) -> DiscreteStateSpace:
        """
        Binomial state space: z ∈ {0, 1, 2, ..., n}
        f(z | a, delta) = C(n,z) * (δ/(1+e^a))^z * (1-δ/(1+e^a))^(n-z)
        """
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial state density function")
        
        n = params['n_trials']
        p_success = delta / (1 + pyo.exp(a))
        
        z_values = list(range(n + 1))  # 0, 1, 2, ..., n
        z_probs = []
        
        for z in z_values:
            prob = comb(n, z) * (p_success ** z) * ((1 - p_success) ** (n - z))
            z_probs.append(prob)
        
        return DiscreteStateSpace(z_values, z_probs)
    
    @staticmethod
    def df_da_binomial_states(a: float, delta: float, params: Dict) -> np.ndarray:
        """
        Derivative of binomial state density with respect to effort a
        df(z|a,δ)/da = -C(n,z) * (δ/(1+e^a))^z * (1-δ/(1+e^a))^(n-z) * 
                       e^a/(1+e^a) * [z - (n-z)*δ/(1+e^a-δ)]
        """
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial state density function")
        
        n = params['n_trials']
        exp_a = pyo.exp(a)
        one_plus_exp_a = 1 + exp_a
        p_success = delta / one_plus_exp_a
        
        derivatives = []
        
        for z in range(n + 1):
            # Base probability
            base_prob = comb(n, z) * (p_success ** z) * ((1 - p_success) ** (n - z))
            
            # Term in brackets
            bracket_term = z - (n - z) * delta / (one_plus_exp_a - delta)
            
            # Full derivative
            deriv = -base_prob * exp_a / one_plus_exp_a * bracket_term
            
            derivatives.append(deriv)
        
        return np.array(derivatives)
