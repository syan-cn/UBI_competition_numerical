from typing import Dict
import pyomo.environ as pyo


class ActionCost:

    @staticmethod
    def power(a: float, theta: float, params: Dict) -> float:
        if 'e_kappa' not in params:
            raise ValueError("Parameter 'e_kappa' is required for power action cost function")
        if 'e_power' not in params:
            raise ValueError("Parameter 'e_power' is required for power action cost function")
        kappa = params['e_kappa']
        power = params['e_power']
        return kappa * theta * (a ** power)
    @staticmethod
    def de_da_power(a: float, theta: float, params: Dict) -> float:
        kappa = params['e_kappa']
        power = params['e_power']
        return kappa * theta * power * (a ** (power - 1))

    @staticmethod
    def d2e_da2_power(a: float, theta: float, params: Dict) -> float:
        kappa = params['e_kappa']
        power = params['e_power']
        return kappa * theta * power * (power - 1) * (a ** (power - 2))

    @staticmethod
    def exponential(a: float, theta: float, params: Dict) -> float:
        if 'e_kappa' not in params:
            raise ValueError("Parameter 'e_kappa' is required for exponential action cost function")
        if 'e_lambda' not in params:
            raise ValueError("Parameter 'e_lambda' is required for exponential action cost function")
        kappa = params['e_kappa']
        lambda_param = params['e_lambda']
        # With bounded actions (0.1, 2.0) and reasonable lambda, exp is safe
        return kappa * theta * (pyo.exp(lambda_param * a) - 1)

    @staticmethod
    def de_da_exponential(a: float, theta: float, params: Dict) -> float:
        kappa = params['e_kappa']
        lambda_param = params['e_lambda']
        return kappa * theta * lambda_param * pyo.exp(lambda_param * a)

    @staticmethod
    def d2e_da2_exponential(a: float, theta: float, params: Dict) -> float:
        kappa = params['e_kappa']
        lambda_param = params['e_lambda']
        return kappa * theta * (lambda_param ** 2) * pyo.exp(lambda_param * a)
