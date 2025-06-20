"""
Probability Functions Module

Contains different functional forms for no accident probability p(a) and their derivatives.
"""

import pyomo.environ as pyo
from typing import Dict
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
        Binomial no accident probability: p(a) = (1 - p̂/(1+e^a))^n
        """
        if 'p_hat' not in params:
            raise ValueError("Parameter 'p_hat' is required for binomial no accident probability function")
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial no accident probability function")
        
        p_hat = params['p_hat']
        n = params['n_trials']
        return (1 - p_hat / (1 + pyo.exp(a))) ** n
    
    @staticmethod
    def dp_da_binomial(a: float, params: Dict) -> float:
        """
        Derivative of binomial no accident probability: 
        dp/da = n*p̂*(1-p̂/(1+e^a))^(n-1) * e^a/(1+e^a)^2
        """
        if 'p_hat' not in params:
            raise ValueError("Parameter 'p_hat' is required for binomial no accident probability function")
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial no accident probability function")
        
        p_hat = params['p_hat']
        n = params['n_trials']
        
        exp_a = pyo.exp(a)
        one_plus_exp_a = 1 + exp_a
        term1 = 1 - p_hat / one_plus_exp_a
        
        return n * p_hat * (term1 ** (n - 1)) * exp_a / (one_plus_exp_a ** 2)
    
    @staticmethod
    def d2p_da2_binomial(a: float, params: Dict) -> float:
        """
        Second derivative of binomial no accident probability:
        d²p/da² = n*p̂*e^a*(1-p̂/(1+e^a))^(n-2) * 
                  [(1-p̂/(1+e^a))*(1-e^a)/(1+e^a)^3 + (n-1)*p̂*e^a/(1+e^a)^4]
        """
        if 'p_hat' not in params:
            raise ValueError("Parameter 'p_hat' is required for binomial no accident probability function")
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial no accident probability function")
        
        p_hat = params['p_hat']
        n = params['n_trials']
        
        exp_a = pyo.exp(a)
        one_plus_exp_a = 1 + exp_a
        term1 = 1 - p_hat / one_plus_exp_a
        
        # First term in brackets
        bracket_term1 = term1 * (1 - exp_a) / (one_plus_exp_a ** 3)
        # Second term in brackets
        bracket_term2 = (n - 1) * p_hat * exp_a / (one_plus_exp_a ** 4)
        
        return n * p_hat * exp_a * (term1 ** (n - 2)) * (bracket_term1 + bracket_term2)

    @staticmethod
    def binomial_logistic(a: float, params: Dict) -> float:
        """
        Binomial logistic no accident probability: 
        p(a) = 1 - sum_{k=0}^{n} (2p̂/(1+exp(-k))) * C(n,k) * exp(a(n-k))/(1+exp(a))^n
        """
        if 'p_hat' not in params:
            raise ValueError("Parameter 'p_hat' is required for binomial logistic no accident probability function")
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial logistic no accident probability function")
        
        p_hat = params['p_hat']
        n = params['n_trials']
        
        exp_a = pyo.exp(a)
        one_plus_exp_a = 1 + exp_a
        one_plus_exp_a_n = one_plus_exp_a ** n
        
        sum_term = 0.0
        for k in range(n + 1):
            coef = 2 * p_hat / (1 + pyo.exp(-k))
            binomial_coef = comb(n, k)
            exp_term = pyo.exp(a * (n - k))
            sum_term += coef * binomial_coef * exp_term / one_plus_exp_a_n
        
        return 1 - sum_term
    
    @staticmethod
    def dp_da_binomial_logistic(a: float, params: Dict) -> float:
        """
        Derivative of binomial logistic no accident probability:
        dp/da = sum_{k=0}^{n} (2p̂/(1+exp(-k))) * C(n,k) * exp(a(n-k))/(1+exp(a))^n * (k - n/(1+exp(a)))
        """
        if 'p_hat' not in params:
            raise ValueError("Parameter 'p_hat' is required for binomial logistic no accident probability function")
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial logistic no accident probability function")
        
        p_hat = params['p_hat']
        n = params['n_trials']
        
        exp_a = pyo.exp(a)
        one_plus_exp_a = 1 + exp_a
        one_plus_exp_a_n = one_plus_exp_a ** n
        
        sum_term = 0.0
        for k in range(n + 1):
            coef = 2 * p_hat / (1 + pyo.exp(-k))
            binomial_coef = comb(n, k)
            exp_term = pyo.exp(a * (n - k))
            bracket_term = k - n / one_plus_exp_a
            sum_term += coef * binomial_coef * exp_term / one_plus_exp_a_n * bracket_term
        
        return sum_term
    
    @staticmethod
    def d2p_da2_binomial_logistic(a: float, params: Dict) -> float:
        """
        Second derivative of binomial logistic no accident probability:
        d²p/da² = sum_{k=0}^{n} (2p̂/(1+exp(-k))) * C(n,k) * exp(a(n-k))/(1+exp(a))^n * 
                  [n*exp(a)/(1+exp(a))² - (k - n/(1+exp(a)))²]
        """
        if 'p_hat' not in params:
            raise ValueError("Parameter 'p_hat' is required for binomial logistic no accident probability function")
        if 'n_trials' not in params:
            raise ValueError("Parameter 'n_trials' is required for binomial logistic no accident probability function")
        
        p_hat = params['p_hat']
        n = params['n_trials']
        
        exp_a = pyo.exp(a)
        one_plus_exp_a = 1 + exp_a
        one_plus_exp_a_n = one_plus_exp_a ** n
        one_plus_exp_a_sq = one_plus_exp_a ** 2
        
        sum_term = 0.0
        for k in range(n + 1):
            coef = 2 * p_hat / (1 + pyo.exp(-k))
            binomial_coef = comb(n, k)
            exp_term = pyo.exp(a * (n - k))
            bracket_term1 = n * exp_a / one_plus_exp_a_sq
            bracket_term2 = (k - n / one_plus_exp_a) ** 2
            bracket_term = bracket_term1 - bracket_term2
            sum_term += coef * binomial_coef * exp_term / one_plus_exp_a_n * bracket_term
        
        return sum_term
