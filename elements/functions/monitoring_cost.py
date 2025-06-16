from typing import Dict

from pyomo import environ as pyo


class MonitoringCost:
    @staticmethod
    def linear(delta: float, params: Dict) -> float:
        if 'c_lambda' not in params:
            raise ValueError("Parameter 'c_lambda' is required for linear monitoring cost function")
        lambda_val = params['c_lambda']
        return lambda_val * delta
    @staticmethod
    def exponential(delta: float, params: Dict) -> float:
        if 'c_lambda' not in params:
            raise ValueError("Parameter 'c_lambda' is required for exponential monitoring cost function")
        lambda_val = params['c_lambda']
        return lambda_val * pyo.exp(delta)
    @staticmethod
    def power(delta: float, params: Dict) -> float:
        if 'c_lambda' not in params:
            raise ValueError("Parameter 'c_lambda' is required for power monitoring cost function")
        if 'c_power' not in params:
            raise ValueError("Parameter 'c_power' is required for power monitoring cost function")
        lambda_val = params['c_lambda']
        power = params['c_power']
        return lambda_val * (delta ** power)
