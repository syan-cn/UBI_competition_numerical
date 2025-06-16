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
    
    @staticmethod
    def exponential(a: float, params: Dict) -> float:
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for exponential no accident probability function")
        alpha = params['p_alpha']
        return 1.0 - pyo.exp(-alpha * a)
    
    @staticmethod
    def dp_da_exponential(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        return alpha * pyo.exp(-alpha * a)
    
    @staticmethod
    def d2p_da2_exponential(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        return -alpha**2 * pyo.exp(-alpha * a)
    
    @staticmethod
    def power(a: float, params: Dict) -> float:
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for power no accident probability function")
        if 'p_beta' not in params:
            raise ValueError("Parameter 'p_beta' is required for power no accident probability function")
        alpha = params['p_alpha']
        beta = params['p_beta']
        return alpha * (a ** beta)
    
    @staticmethod
    def dp_da_power(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        return alpha * beta * (a ** (beta - 1))
    
    @staticmethod
    def d2p_da2_power(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        return alpha * beta * (beta - 1) * (a ** (beta - 2))
    
    @staticmethod
    def logistic(a: float, params: Dict) -> float:
        if 'p_alpha' not in params:
            raise ValueError("Parameter 'p_alpha' is required for logistic no accident probability function")
        if 'p_beta' not in params:
            raise ValueError("Parameter 'p_beta' is required for logistic no accident probability function")
        alpha = params['p_alpha']
        beta = params['p_beta']
        return 1.0 / (1.0 + pyo.exp(-alpha * (a - beta)))
    
    @staticmethod
    def dp_da_logistic(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        p = 1.0 / (1.0 + pyo.exp(-alpha * (a - beta)))
        return alpha * p * (1.0 - p)
    
    @staticmethod
    def d2p_da2_logistic(a: float, params: Dict) -> float:
        alpha = params['p_alpha']
        beta = params['p_beta']
        p = 1.0 / (1.0 + pyo.exp(-alpha * (a - beta)))
        return alpha**2 * p * (1.0 - p) * (1.0 - 2.0 * p) 