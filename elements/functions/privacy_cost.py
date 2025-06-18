from typing import Dict

from pyomo import environ as pyo


class PrivacyCost:

    @staticmethod
    def linear(delta: float, params: Dict) -> float:
        if 'm_gamma' not in params:
            raise ValueError("Parameter 'm_gamma' is required for linear privacy cost function")
        gamma = params['m_gamma']
        return gamma * delta

    @staticmethod
    def exponential(delta: float, params: Dict) -> float:
        if 'm_gamma' not in params:
            raise ValueError("Parameter 'm_gamma' is required for exponential privacy cost function")
        gamma = params['m_gamma']
        return gamma * pyo.exp(delta)

    @staticmethod
    def power(delta: float, params: Dict) -> float:
        if 'm_gamma' not in params:
            raise ValueError("Parameter 'm_gamma' is required for power privacy cost function")
        if 'm_power' not in params:
            raise ValueError("Parameter 'm_power' is required for power privacy cost function")
        gamma = params['m_gamma']
        power = params['m_power']
        return gamma * (delta ** power)
