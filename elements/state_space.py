"""
Discrete State Space Module

This module contains classes and functions for handling discrete state spaces
in insurance models, including state density functions.
"""

import numpy as np
from typing import Dict, Tuple, List


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