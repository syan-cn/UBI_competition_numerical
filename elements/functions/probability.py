"""
Probability Functions Module

Contains different functional forms for no accident probability p(a) and their derivatives.
"""

import pyomo.environ as pyo
from typing import Dict
import numpy as np
from scipy.special import comb

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
    def binomial(a: float, params: Dict) -> float:
        """
        Binomial accident probability: p(a) = 1 - [1 - σ(a)p̂]^n
        where σ(a) = 1 - a
        """
        if 'p_hat' not in params:
            raise ValueError("Parameter 'p_hat' is required for binomial no accident probability function")
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial no accident probability function")
        
        p_hat = params['p_hat']
        n = params['n_trials']
        
        return 1 - (1 - (1 - a) * p_hat) ** n
    
    @staticmethod
    def dp_da_binomial(a: float, params: Dict) -> float:
        """
        Derivative of binomial accident probability: dp/da = -n*p̂*[1-(1-a)*p̂]^(n-1)
        """
        if 'p_hat' not in params:
            raise ValueError("Parameter 'p_hat' is required for binomial no accident probability function")
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial no accident probability function")
        
        p_hat = params['p_hat']
        n = params['n_trials']
        
        return -n * p_hat * (1 - (1 - a) * p_hat) ** (n - 1)
    
    @staticmethod
    def d2p_da2_binomial(a: float, params: Dict) -> float:
        """
        Second derivative of binomial accident probability:
        d²p/da² = -n(n-1)p_hat² * [1-(1-a)p_hat]^(n-2)
        """
        if 'p_hat' not in params:
            raise ValueError("Parameter 'p_hat' is required for binomial no accident probability function")
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial no accident probability function")
        
        p_hat = params['p_hat']
        n = params['n_trials']
        
        return -n * (n - 1) * (p_hat ** 2) * (1 - (1 - a) * p_hat) ** (n - 2)
