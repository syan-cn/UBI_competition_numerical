"""
Flexible Functions Module

Contains the Functions class that selects and delegates to the appropriate function implementations.
"""

from typing import Dict
from elements.functions.probability import NoAccidentProbability
from elements.functions.action_cost import ActionCost
from elements.functions.privacy_cost import PrivacyCost
from elements.functions.utility import Utility
from elements.state_space import DiscreteStateSpace, StateDensity
import numpy as np
import pyomo.environ as pyo


class FunctionTemplates:
    """Abstract function templates for the insurance model."""

    @staticmethod
    def p(a: float, params: Dict) -> float:
        """
        No accident probability function.

        Args:
            a: Action level
            params: Dictionary containing function parameters

        Returns:
            Probability of no accident
        """
        raise NotImplementedError("Subclass must implement p(a, params)")

    @staticmethod
    def m(delta: float, params: Dict) -> float:
        """
        Monitoring cost function.

        Args:
            delta: Monitoring level
            params: Dictionary containing function parameters

        Returns:
            Monitoring cost
        """
        raise NotImplementedError("Subclass must implement m(delta, params)")

    @staticmethod
    def e(a: float, theta: float, params: Dict) -> float:
        """
        Action cost function.

        Args:
            a: Action level
            theta: Driver's risk type
            params: Dictionary containing function parameters

        Returns:
            Action cost
        """
        raise NotImplementedError("Subclass must implement e(a, theta, params)")

    @staticmethod
    def u(x: float, params: Dict) -> float:
        """
        Utility function.

        Args:
            x: Wealth level
            params: Dictionary containing function parameters

        Returns:
            Utility value
        """
        raise NotImplementedError("Subclass must implement u(x, params)")

    @staticmethod
    def f(a: float, delta: float, params: Dict) -> DiscreteStateSpace:
        """
        State density function - now returns DiscreteStateSpace.

        Args:
            a: Action level
            delta: Monitoring level
            params: Dictionary containing function parameters

        Returns:
            DiscreteStateSpace object
        """
        raise NotImplementedError("Subclass must implement f(a, delta, params)")

    @staticmethod
    def c(delta: float, params: Dict) -> float:
        """
        Insurer's cost function.

        Args:
            delta: Monitoring level
            params: Dictionary containing function parameters

        Returns:
            Insurer's cost
        """
        raise NotImplementedError("Subclass must implement c(delta, params)")


class Functions(FunctionTemplates):
    """Flexible function class that can use different functional forms."""

    def __init__(self, function_config: Dict):
        self.function_config = function_config
        required_keys = ['p', 'm', 'e', 'u', 'f', 'c']
        for key in required_keys:
            if key not in function_config:
                raise ValueError(f"Function configuration must include '{key}'")
        valid_forms = {
            'p': ['linear', 'binomial'],
            'm': ['linear', 'exponential', 'power'],
            'e': ['power'],
            'u': ['exponential', 'power', 'logarithmic'],
            'f': ['binary_states', 'binomial_states'],
            'c': ['linear', 'exponential', 'power']
        }
        for key, form in function_config.items():
            if form not in valid_forms[key]:
                raise ValueError(f"Invalid functional form '{form}' for '{key}'. Valid forms: {valid_forms[key]}")

    @staticmethod
    def choice_probabilities(V0, V1, V2, mu):
        exp_V0 = pyo.exp(mu * V0)
        exp_V1 = pyo.exp(mu * V1)
        exp_V2 = pyo.exp(mu * V2)
        denominator = exp_V0 + exp_V1 + exp_V2
        prob_0 = exp_V0 / denominator
        prob_1 = exp_V1 / denominator
        prob_2 = exp_V2 / denominator
        return prob_0, prob_1, prob_2

    def p(self, a: float, params: Dict) -> float:
        form = self.function_config['p']
        if form == 'linear':
            return NoAccidentProbability.linear(a, params)
        elif form == 'binomial':
            return NoAccidentProbability.binomial(a, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def m(self, delta: float, params: Dict) -> float:
        form = self.function_config['m']
        if form == 'linear':
            return PrivacyCost.linear(delta, params)
        elif form == 'exponential':
            return PrivacyCost.exponential(delta, params)
        elif form == 'power':
            return PrivacyCost.power(delta, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def e(self, a: float, theta: float, params: Dict) -> float:
        form = self.function_config['e']
        if form == 'power':
            return ActionCost.power(a, theta, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def u(self, x: float, params: Dict) -> float:
        form = self.function_config['u']
        if form == 'exponential':
            return Utility.exponential(x, params)
        elif form == 'power':
            return Utility.power(x, params)
        elif form == 'logarithmic':
            return Utility.logarithmic(x, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def f(self, a: float, delta: float, params: Dict) -> DiscreteStateSpace:
        form = self.function_config['f']
        if form == 'binary_states':
            return StateDensity.binary_states(a, delta, params)
        elif form == 'binomial_states':
            return StateDensity.binomial_states(a, delta, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def c(self, delta: float, params: Dict) -> float:
        form = self.function_config['c']
        if form == 'linear':
            return PrivacyCost.linear(delta, params)
        elif form == 'exponential':
            return PrivacyCost.exponential(delta, params)
        elif form == 'power':
            return PrivacyCost.power(delta, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def dp_da(self, a: float, params: Dict) -> float:
        form = self.function_config['p']
        if form == 'linear':
            return NoAccidentProbability.dp_da_linear(a, params)
        elif form == 'binomial':
            return NoAccidentProbability.dp_da_binomial(a, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def d2p_da2(self, a: float, params: Dict) -> float:
        form = self.function_config['p']
        if form == 'linear':
            return NoAccidentProbability.d2p_da2_linear(a, params)
        elif form == 'binomial':
            return NoAccidentProbability.d2p_da2_binomial(a, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def de_da(self, a: float, theta: float, params: Dict) -> float:
        form = self.function_config['e']
        if form == 'power':
            return ActionCost.de_da_power(a, theta, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def d2e_da2(self, a: float, theta: float, params: Dict) -> float:
        form = self.function_config['e']
        if form == 'power':
            return ActionCost.d2e_da2_power(a, theta, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def df_da(self, a: float, delta: float, params: Dict) -> np.ndarray:
        form = self.function_config['f']
        if form == 'binary_states':
            return StateDensity.df_da_binary_states(a, delta, params)
        elif form == 'binomial_states':
            return StateDensity.df_da_binomial_states(a, delta, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def du_dx(self, x: float, params: Dict) -> float:
        form = self.function_config['u']
        if form == 'exponential':
            return Utility.du_dx_exponential(x, params)
        elif form == 'power':
            return Utility.du_dx_power(x, params)
        elif form == 'logarithmic':
            return Utility.du_dx_logarithmic(x, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")

    def d2u_dx2(self, x: float, params: Dict) -> float:
        form = self.function_config['u']
        if form == 'exponential':
            return Utility.d2u_dx2_exponential(x, params)
        elif form == 'power':
            return Utility.d2u_dx2_power(x, params)
        elif form == 'logarithmic':
            return Utility.d2u_dx2_logarithmic(x, params)
        else:
            raise ValueError(f"Unknown functional form: {form}")
