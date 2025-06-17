"""
Probability Functions Module

Contains different functional forms for no accident probability p(a) and their derivatives.
"""

import pyomo.environ as pyo
from typing import Dict

class NoAccidentProbability:
    """Different functional forms for no accident probability p(a)."""
    
    @staticmethod
    def linear(a: float, params: Dict) -> float:
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for linear no accident probability function")
        if 'p_beta' not in params:
            raise ValueError("Parameter 'p_beta' is required for linear no accident probability function")
        alpha = params['p_alpha']
        beta = params['p_beta']
        return alpha + beta * a
    
    @staticmethod
    def dp_da_linear(a: float, params: Dict) -> float:
        return params['p_beta']
    
    @staticmethod
    def d2p_da2_linear(a: float, params: Dict) -> float:
        return 0.0
