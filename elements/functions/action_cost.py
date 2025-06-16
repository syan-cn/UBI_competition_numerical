from typing import Dict


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
