"""
Utility Functions Module

Contains different functional forms for utility u(x) and their derivatives.
"""

import numpy as np
import pyomo.environ as pyo
from typing import Dict

class Utility:

    @staticmethod
    def exponential(x: float, params: Dict) -> float:
        if 'u_rho' not in params:
            raise ValueError("Parameter 'u_rho' is required for exponential utility function")
        if 'u_max_val' not in params:
            raise ValueError("Parameter 'u_max_val' is required for exponential utility function")
        rho = params['u_rho']
        max_val = params['u_max_val']
        return max_val * (1.0 - pyo.exp(-rho * x))

    @staticmethod
    def du_dx_exponential(x: float, params: Dict) -> float:
        rho = params['u_rho']
        max_val = params['u_max_val']
        return max_val * rho * pyo.exp(-rho * x)

    @staticmethod
    def d2u_dx2_exponential(x: float, params: Dict) -> float:
        rho = params['u_rho']
        max_val = params['u_max_val']
        return -max_val * rho**2 * pyo.exp(-rho * x)

    @staticmethod
    def power(x: float, params: Dict) -> float:
        if 'u_gamma' not in params:
            raise ValueError("Parameter 'u_gamma' is required for power utility function")
        gamma = params['u_gamma']
        if gamma == 1:
            return x
        else:
            return (x ** gamma) / gamma

    @staticmethod
    def du_dx_power(x: float, params: Dict) -> float:
        gamma = params['u_gamma']
        if gamma == 1:
            return 1.0
        else:
            return x ** (gamma - 1)

    @staticmethod
    def d2u_dx2_power(x: float, params: Dict) -> float:
        gamma = params['u_gamma']
        if gamma == 1:
            return 0.0
        else:
            return (gamma - 1) * x ** (gamma - 2)

    @staticmethod
    def logarithmic(x: float, params: Dict) -> float:
        return np.log(x)

    @staticmethod
    def du_dx_logarithmic(x: float, params: Dict) -> float:
        return 1.0 / x

    @staticmethod
    def d2u_dx2_logarithmic(x: float, params: Dict) -> float:
        return -1.0 / (x ** 2)
