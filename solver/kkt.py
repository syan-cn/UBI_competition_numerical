"""
DuopolySolverKKT: Self-contained KKT-based duopoly insurance solver

This module defines the DuopolySolverKKT class, which encapsulates all logic, parameters, and helper methods
required to solve the duopoly insurance equilibrium problem using KKT conditions and Pyomo optimization.

No dependency on helper.py. All logic is self-contained.
"""

import numpy as np
import pyomo.environ as pyo
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize
from pyomo.opt import SolverStatus

class DuopolySolverKKT:
    """
    Self-contained numerical solver for the duopoly insurance model using KKT conditions.
    All logic and helper methods are encapsulated in this class.
    """
    def __init__(self, functions, params):
        self.functions = functions
        self.params = params
        self.function_config = functions.function_config  # Store function configuration
        # Extract key parameters
        self.W = params['W']
        self.s = params['s']
        self.N = params['N']
        self.delta1 = params['delta1']
        self.delta2 = params['delta2']
        self.mu = params['mu']
        self.theta_min = params['theta_min']
        self.theta_max = params['theta_max']
        self.n_theta = params['n_theta']
        # Create risk type grid
        self.theta_grid = np.linspace(self.theta_min, self.theta_max, self.n_theta)
        # Risk type distribution (uniform by default)
        self.h_theta = np.ones(self.n_theta) / self.n_theta
        # Get state space for reference (will be computed dynamically)
        state_space = self.functions.f(1.0, 1.0, self.params)
        self.z_values, z_probs = state_space.get_all_states()
        self.z_states = len(self.z_values)

    def build_model(self, solver_name='ipopt', verbose=False, debug_mode=True, executable_path=None, save_model=False, model_filename=None):
        """
        Solve the duopoly equilibrium using KKT conditions with pyomo.

        Args:
            solver_name: Name of the solver to use
            verbose: Whether to print detailed output
            debug_mode: Whether to enable debug mode
            executable_path: Optional path to solver executable
            save_model: Whether to save the model in .nl format
            model_filename: Optional filename for the saved model (without extension)
        """
        print("Setting up KKT system for duopoly equilibrium...")
        
        # Create the optimization model
        model = pyo.ConcreteModel()
        
        # Sets
        model.THETA = pyo.Set(initialize=range(self.n_theta))  # Risk types
        model.Z = pyo.Set(initialize=range(self.z_states))     # States
        model.I = pyo.Set(initialize=[1, 2])                  # Insurers
        
        # Parameters
        model.W = pyo.Param(initialize=self.W)
        model.s = pyo.Param(initialize=self.s)
        model.N = pyo.Param(initialize=self.N)
        model.theta_vals = pyo.Param(model.THETA, initialize={i: self.theta_grid[i] for i in range(self.n_theta)})
        model.h_theta = pyo.Param(model.THETA, initialize={i: self.h_theta[i] for i in range(self.n_theta)})
        model.delta1 = pyo.Param(initialize=self.delta1)
        model.delta2 = pyo.Param(initialize=self.delta2)
        
        # Decision Variables with VERY tight bounds to prevent any overflow
        model.a = pyo.Var(model.I, model.THETA, domain=pyo.Reals, bounds=(0.0, 1.0))
        model.phi1 = pyo.Var(model.I, domain=pyo.NonNegativeReals)
        model.phi2 = pyo.Var(model.I, model.Z, domain=pyo.NonNegativeReals)

        # Lagrange multipliers
        model.lam = pyo.Var(model.I, model.THETA, domain=pyo.Reals)
        
        # Wealth constraints to ensure non-negative wealth
        def wealth_no_accident_constraint_rule(model, i):
            """Constraint to ensure W - φ₁^i ≥ 0 (non-negative wealth after premium payment)."""
            return model.W - model.phi1[i] >= 0
        
        model.wealth_no_accident_constraint = pyo.Constraint(model.I, rule=wealth_no_accident_constraint_rule)
        
        def wealth_accident_constraint_rule(model, i, z):
            """Constraint to ensure W - φ₁^i + φ₂^i(z) - s ≥ 0 (non-negative wealth after accident with indemnity)."""
            return model.W - model.phi1[i] + model.phi2[i, z] - model.s >= 0
        
        model.wealth_accident_constraint = pyo.Constraint(model.I, model.Z, rule=wealth_accident_constraint_rule)
        
        def incentive_constraint_rule(model, i, t):
            """Incentive compatibility constraint G(θ) = 0."""
            theta = model.theta_vals[t]
            a_val = model.a[i, t]
            phi1_val = model.phi1[i]
            delta_val = model.delta1 if i == 1 else model.delta2
            
            # Get phi2 values for all states
            phi2_values = np.array([model.phi2[i, z] for z in model.Z])
            
            # Use the existing compute_G_function method
            G_val = self.compute_G_function(theta, a_val, phi1_val, phi2_values, delta_val)
            
            return G_val == 0
        
        model.incentive_constraint = pyo.Constraint(model.I, model.THETA, rule=incentive_constraint_rule)
        
        # Stationarity conditions
        def stationarity_action_rule(model, i, t):
            """Stationarity with respect to action a^i(θ)."""
            theta = model.theta_vals[t]
            a_val = model.a[i, t]
            phi1_val = model.phi1[i]
            delta_val = model.delta1 if i == 1 else model.delta2
            phi2_values = np.array([model.phi2[i, z] for z in model.Z])
            
            other_insurer = 2 if i == 1 else 1
            a_other = model.a[other_insurer, t]
            phi1_other = model.phi1[other_insurer]
            phi2_other_values = np.array([model.phi2[other_insurer, z] for z in model.Z])

            # Use existing helper functions to compute choice probabilities
            P0, P1, P2 = self.compute_choice_probabilities(theta, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
            Pi = P1 if i == 1 else P2

            dPi_da = self.compute_dPi_da(theta, i, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)

            dG_da = self.compute_dG_da(theta, a_val, phi1_val, phi2_values, delta_val)
            
            # Compute profit term: φ₁^i - (1-p(a^i(θ))) ∫ φ₂^i(z) f(z|a^i(θ),δ^i) dz
            p_no_accident = self.functions.p(a_val, self.params)
            p_accident = 1 - p_no_accident
            
            # Get state space and compute integral
            state_space = self.functions.f(a_val, delta_val, self.params)
            z_values, z_probs = state_space.get_all_states()
            
            integral_phi2_f = 0.0
            for j, z in enumerate(z_values):
                integral_phi2_f += phi2_values[j] * z_probs[j]
            
            profit_term = phi1_val - p_accident * integral_phi2_f
            
            # Compute second term bracket: -p'(a^i(θ)) ∫ φ₂^i(z) f(z|a^i(θ),δ^i) dz + (1-p(a^i(θ))) ∫ φ₂^i(z) ∂f(z|a^i(θ),δ^i)/∂a^i(θ) dz
            dp_da = self.functions.dp_da(a_val, self.params)
            df_da = self.functions.df_da(a_val, delta_val, self.params)
            
            integral_phi2_df_da = 0.0
            for j, z in enumerate(z_values):
                integral_phi2_df_da += phi2_values[j] * df_da[j]
            
            second_term_bracket = -dp_da * integral_phi2_f + p_accident * integral_phi2_df_da
            
            # Complete KKT stationarity condition
            term1 = self.N * dPi_da * profit_term
            term2 = -self.N * Pi * second_term_bracket
            term3 = model.lam[i, t] * dG_da
            
            return term1 + term2 + term3 == 0
        
        model.stationarity_action = pyo.Constraint(model.I, model.THETA, rule=stationarity_action_rule)
        
        def stationarity_premium_rule(model, i):
            """Stationarity with respect to premium φ₁^i."""
            # Get monitoring level for this insurer
            delta_val = model.delta1 if i == 1 else model.delta2
            
            # Calculate complete integral over all risk types
            integral_term = 0.0
            lambda_dG_integral = 0.0
            
            for t in model.THETA:
                theta = model.theta_vals[t]
                h_theta_val = model.h_theta[t]
                a_val = model.a[i, t]
                phi1_val = model.phi1[i]
                
                # Get phi2 values for all states
                phi2_values = np.array([model.phi2[i, z] for z in model.Z])
                
                # Get contract values for the other insurer
                other_insurer = 2 if i == 1 else 1
                a_other = model.a[other_insurer, t]
                phi1_other = model.phi1[other_insurer]
                phi2_other_values = np.array([model.phi2[other_insurer, z] for z in model.Z])
                
                # Use existing helper functions
                P0, P1, P2 = self.compute_choice_probabilities(theta, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
                Pi = P1 if i == 1 else P2
                
                dPi_dphi1 = self.compute_dPi_dphi1(theta, i, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
                dG_dphi1 = self.compute_dG_dphi1(theta, a_val, phi1_val, phi2_values, delta_val)
                
                # Compute profit term: φ₁^i - (1-p(a^i(θ))) ∫ φ₂^i(z) f(z|a^i(θ),δ^i) dz
                p_no_accident = self.functions.p(a_val, self.params)
                p_accident = 1 - p_no_accident
                
                # Get state space and compute integral
                state_space = self.functions.f(a_val, delta_val, self.params)
                z_values, z_probs = state_space.get_all_states()
                
                integral_phi2_f = 0.0
                for j, z in enumerate(z_values):
                    integral_phi2_f += phi2_values[j] * z_probs[j]
                
                profit_term = phi1_val - p_accident * integral_phi2_f
                
                # First integral term: ∫ h(θ) [∂P_i(θ)/∂φ₁^i * profit_term + P_i(θ)] dθ
                integral_term += h_theta_val * (dPi_dphi1 * profit_term + Pi)
                
                # Second integral term: ∫ λ(θ) ∂G(θ)/∂φ₁^i h(θ) dθ
                lambda_dG_integral += model.lam[i, t] * dG_dphi1 * h_theta_val
            
            # Stationarity condition
            return self.N * integral_term + lambda_dG_integral == 0
        
        model.stationarity_premium = pyo.Constraint(model.I, rule=stationarity_premium_rule)
        
        def stationarity_indemnity_rule(model, i, z):
            """Stationarity with respect to indemnity φ₂^i(z)."""
            # Get monitoring level for this insurer
            delta_val = model.delta1 if i == 1 else model.delta2
            
            # Calculate complete integrals over all risk types
            first_integral = 0.0     # ∫ h(θ) P_i(θ) (1-p(a^i(θ))) f(z|a^i(θ),δ^i) dθ
            second_integral = 0.0    # ∫ h(θ) ∂P_i(θ)/∂φ₂^i(z) [profit_term] dθ
            lambda_dG_integral = 0.0 # ∫ λ(θ) ∂G(θ)/∂φ₂^i(z) h(θ) dθ
            
            for t in model.THETA:
                theta = model.theta_vals[t]
                h_theta_val = model.h_theta[t]
                a_val = model.a[i, t]
                phi1_val = model.phi1[i]
                
                # Get phi2 values for all states
                phi2_values = np.array([model.phi2[i, z_idx] for z_idx in model.Z])
                
                # Get contract values for the other insurer
                other_insurer = 2 if i == 1 else 1
                a_other = model.a[other_insurer, t]
                phi1_other = model.phi1[other_insurer]
                phi2_other_values = np.array([model.phi2[other_insurer, z_idx] for z_idx in model.Z])
                
                # Use existing helper functions
                P0, P1, P2 = self.compute_choice_probabilities(theta, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
                Pi = P1 if i == 1 else P2
                
                dPi_dphi2 = self.compute_dPi_dphi2(theta, i, z, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
                dG_dphi2 = self.compute_dG_dphi2(theta, a_val, phi1_val, phi2_values, delta_val, z)
                
                # Compute profit term: φ₁^i - (1-p(a^i(θ))) ∫ φ₂^i(t) f(t|a^i(θ),δ^i) dt
                p_no_accident = self.functions.p(a_val, self.params)
                p_accident = 1 - p_no_accident
                
                # Get state space and compute integral
                state_space = self.functions.f(a_val, delta_val, self.params)
                z_values, z_probs = state_space.get_all_states()
                
                integral_phi2_f = 0.0
                for j, z_val in enumerate(z_values):
                    integral_phi2_f += phi2_values[j] * z_probs[j]
                
                profit_term = phi1_val - p_accident * integral_phi2_f
                
                # Get f(z|a^i(θ),δ^i) for the specific state z
                f_z_val = z_probs[z]
                
                # First integral: ∫ h(θ) P_i(θ) (1-p(a^i(θ))) f(z|a^i(θ),δ^i) dθ
                first_integral += h_theta_val * Pi * p_accident * f_z_val
                
                # Second integral: ∫ h(θ) ∂P_i(θ)/∂φ₂^i(z) [profit_term] dθ
                second_integral += h_theta_val * dPi_dphi2 * profit_term
                
                # Third integral: ∫ λ(θ) ∂G(θ)/∂φ₂^i(z) h(θ) dθ
                lambda_dG_integral += model.lam[i, t] * dG_dphi2 * h_theta_val
            
            # Stationarity condition
            return -self.N * first_integral + self.N * second_integral + lambda_dG_integral == 0
        
        model.stationarity_indemnity = pyo.Constraint(model.I, model.Z, rule=stationarity_indemnity_rule)
        
        # Objective function: minimize the sum of squared violations of KKT conditions
        # This creates a meaningful objective for multistart optimization
        def kkt_violation_objective(model):
            """Objective function that measures violation of KKT conditions."""
            total_violation = 0.0
            
            # Incentive compatibility constraint violations
            for i in model.I:
                for t in model.THETA:
                    theta = model.theta_vals[t]
                    a_val = model.a[i, t]
                    phi1_val = model.phi1[i]
                    delta_val = model.delta1 if i == 1 else model.delta2
                    phi2_values = np.array([model.phi2[i, z] for z in model.Z])
                    
                    G_val = self.compute_G_function(theta, a_val, phi1_val, phi2_values, delta_val)
                    total_violation += G_val**2
            
            # # Stationarity condition violations (simplified measure)
            # for i in model.I:
            #     for t in model.THETA:
            #         # Add small penalty for non-zero Lagrange multipliers when constraints are satisfied
            #         total_violation += 0.1 * model.lam[i, t]**2
            
            # # Add regularization terms to prevent extreme values
            # for i in model.I:
            #     # Penalize extreme premium values
            #     total_violation += 0.01 * (model.phi1[i] - self.W/2)**2
                
            #     for t in model.THETA:
            #         # Penalize extreme action values
            #         total_violation += 0.01 * (model.a[i, t] - 0.5)**2
                    
            #         for z in model.Z:
            #             # Penalize extreme indemnity values
            #             total_violation += 0.01 * (model.phi2[i, z] - self.W/2)**2
            
            return total_violation / 1e5
        
        model.obj = pyo.Objective(rule=kkt_violation_objective, sense=pyo.minimize)

        return model

    def compute_reservation_utility(self, theta: float) -> float:
        """Compute reservation utility V_0(theta) using Pyomo optimization."""

        # Create Pyomo model for reservation utility optimization
        reservation_model = pyo.ConcreteModel()

        # Decision variable: action level
        reservation_model.a = pyo.Var(domain=pyo.Reals)

        # Objective function: maximize expected utility
        def objective_rule(reservation_model):
            a_val = reservation_model.a
            p_no_accident = self.functions.p(a_val, self.params)
            p_accident = 1 - p_no_accident
            e_val = self.functions.e(a_val, theta, self.params)

            utility_no_accident = self.functions.u(self.W, self.params)
            utility_accident = self.functions.u(self.W - self.s, self.params)

            expected_utility = p_no_accident * utility_no_accident + p_accident * utility_accident - e_val
            return -expected_utility  # Minimize negative utility (maximize utility)

        reservation_model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        # Solve the optimization problem
        solver = pyo.SolverFactory('ipopt')
        solver.solve(reservation_model, tee=False)
        
        optimal_utility = -pyo.value(reservation_model.obj)  # Convert back to positive utility
        return optimal_utility / 10

    def compute_expected_utility(self, a: float, phi1: float, phi2_values: np.ndarray, delta: float,
                                 theta: float) -> float:
        """
        Compute expected utility for given contract using discrete indemnity values.

        Args:
            a: Action level
            phi1: Premium
            phi2_values: Array of indemnity values for each state (length n_states)
            delta: Monitoring level
            theta: Risk type
        """
        p_no_accident = self.functions.p(a, self.params)  # Probability of no accident
        p_accident = 1 - p_no_accident  # Probability of accident
        e_val = self.functions.e(a, theta, self.params)

        # Get discrete state space for this action and monitoring level
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()

        # Expected utility when accident occurs (sum over discrete states)
        expected_utility_accident = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_val = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            expected_utility_accident += u_val * z_probs[i]

        # Expected utility when no accident occurs
        utility_no_accident = self.functions.u(self.W - phi1, self.params)
        m_val = self.functions.m(delta, self.params)

        return p_no_accident * utility_no_accident + p_accident * expected_utility_accident - e_val - m_val

    def compute_utility_Vi(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """
        Compute utility V_i(θ) for a given consumer type θ and contract (a, φ1, φ2).

        Mathematical formulation:
        V_k(θ) = p(a^k(θ)) u(W-φ₁^k) + [1-p(a^k(θ))] ∫ u(W-φ₁^k+φ₂^k(z)-s) f(z|a^k(θ),δ^k) dz - e(a^k(θ),θ) - m(δ^k)
        """
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        e_val = self.functions.e(a, theta, self.params)
        m_val = self.functions.m(delta, self.params)

        # Get state space
        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()

        # Utility when no accident occurs
        utility_no_accident = self.functions.u(self.W - phi1, self.params)

        # Expected utility when accident occurs
        expected_utility_accident = 0.0
        for i, z in enumerate(z_values):
            u_accident = self.functions.u(self.W - phi1 + phi2_values[i] - self.s, self.params)
            expected_utility_accident += u_accident * z_probs[i]

        return p_no_accident * utility_no_accident + p_accident * expected_utility_accident - e_val - m_val

    def compute_choice_probabilities(self, theta: float,
                                     a1: float, phi1_1: float, phi2_1: np.ndarray,
                                     a2: float, phi1_2: float, phi2_2: np.ndarray) -> tuple[float, float, float]:
        """Compute choice probabilities P_0(θ), P_1(θ), P_2(θ) using multinomial logit."""
        V0 = self.compute_reservation_utility(theta)
        V1 = self.compute_utility_Vi(theta, a1, phi1_1, phi2_1, self.delta1)
        V2 = self.compute_utility_Vi(theta, a2, phi1_2, phi2_2, self.delta2)

        utilities = [V0, V1, V2]
        scaled_utilities = [u / self.mu for u in utilities]

        # Compute choice probabilities using Pyomo exp
        exp_utilities = [pyo.exp(u) for u in scaled_utilities]
        denom = sum(exp_utilities)

        P0 = exp_utilities[0] / denom
        P1 = exp_utilities[1] / denom
        P2 = exp_utilities[2] / denom

        return P0, P1, P2

    def compute_dPi_da(self, theta: float, i: int,
                       a1: float, phi1_1: float, phi2_1: np.ndarray,
                       a2: float, phi1_2: float, phi2_2: np.ndarray) -> float:
        """Compute derivative of choice probability P_i with respect to action a_i."""
        P0, P1, P2 = self.compute_choice_probabilities(theta, a1, phi1_1, phi2_1, a2, phi1_2, phi2_2)

        if i == 1:
            dVi_da = self.compute_G_function(theta, a1, phi1_1, phi2_1, self.delta1)
            Pi = P1
        else:
            dVi_da = self.compute_G_function(theta, a2, phi1_2, phi2_2, self.delta2)
            Pi = P2

        return (1 / self.mu) * Pi * (1 - Pi) * dVi_da

    def compute_dVi_da(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute derivative of V_i with respect to action a."""
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident
        dp_da = self.functions.dp_da(a, self.params)
        de_da = self.functions.de_da(a, theta, self.params)

        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)

        utility_no_accident = self.functions.u(self.W - phi1, self.params)

        integral1 = 0.0
        integral2 = 0.0

        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            integral1 += u_accident * z_probs[i]
            integral2 += u_accident * df_da[i]

        return dp_da * (utility_no_accident - integral1) + p_accident * integral2 - de_da

    def compute_dVi_dphi1(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute derivative of V_i with respect to premium φ₁."""
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident

        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()

        du_dphi1_no_accident = -self.functions.du_dx(self.W - phi1, self.params)

        integral_du_dphi1_accident = 0.0
        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            du_dphi1_accident = -self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
            integral_du_dphi1_accident += du_dphi1_accident * z_probs[i]

        return p_no_accident * du_dphi1_no_accident + p_accident * integral_du_dphi1_accident

    def compute_dVi_dphi2(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float,
                          z_idx: int) -> float:
        """Compute derivative of V_i with respect to indemnity φ₂(z)."""
        p_accident = 1 - self.functions.p(a, self.params)

        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()

        phi2_val = phi2_values[z_idx]
        du_dphi2 = self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
        f_z = z_probs[z_idx]

        return p_accident * du_dphi2 * f_z

    def compute_dPi_dphi1(self, theta: float, i: int,
                          a1: float, phi1_1: float, phi2_1: np.ndarray,
                          a2: float, phi1_2: float, phi2_2: np.ndarray) -> float:
        """Compute derivative of choice probability P_i with respect to premium φ₁^i."""
        P0, P1, P2 = self.compute_choice_probabilities(theta, a1, phi1_1, phi2_1, a2, phi1_2, phi2_2)

        if i == 1:
            dVi_dphi1 = self.compute_dVi_dphi1(theta, a1, phi1_1, phi2_1, self.delta1)
            Pi = P1
        else:
            dVi_dphi1 = self.compute_dVi_dphi1(theta, a2, phi1_2, phi2_2, self.delta2)
            Pi = P2

        return (1 / self.mu) * Pi * (1 - Pi) * dVi_dphi1

    def compute_dPi_dphi2(self, theta: float, i: int, z_idx: int,
                          a1: float, phi1_1: float, phi2_1: np.ndarray,
                          a2: float, phi1_2: float, phi2_2: np.ndarray) -> float:
        """Compute derivative of choice probability P_i with respect to indemnity φ₂^i(z)."""
        P0, P1, P2 = self.compute_choice_probabilities(theta, a1, phi1_1, phi2_1, a2, phi1_2, phi2_2)

        if i == 1:
            dVi_dphi2 = self.compute_dVi_dphi2(theta, a1, phi1_1, phi2_1, self.delta1, z_idx)
            Pi = P1
        else:
            dVi_dphi2 = self.compute_dVi_dphi2(theta, a2, phi1_2, phi2_2, self.delta2, z_idx)
            Pi = P2

        return (1 / self.mu) * Pi * (1 - Pi) * dVi_dphi2

    def compute_G_function(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute the G(θ) function from the mathematical model."""
        dp_da = self.functions.dp_da(a, self.params)
        de_da = self.functions.de_da(a, theta, self.params)
        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident

        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)

        utility_no_accident = self.functions.u(self.W - phi1, self.params)

        integral_1 = 0.0
        integral_2 = 0.0

        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)
            integral_1 += u_accident * z_probs[i]
            integral_2 += u_accident * df_da[i]

        return dp_da * (utility_no_accident - integral_1) + p_accident * integral_2 - de_da

    def compute_dG_dphi1(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute derivative of G(θ) with respect to φ₁."""
        dp_da = self.functions.dp_da(a, self.params)
        p_accident = 1 - self.functions.p(a, self.params)

        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)

        du_no_accident = self.functions.du_dx(self.W - phi1, self.params)

        integral_marginal_1 = 0.0
        integral_marginal_2 = 0.0

        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            du_accident = self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
            integral_marginal_1 += du_accident * z_probs[i]
            integral_marginal_2 += du_accident * df_da[i]

        return -dp_da * (du_no_accident - integral_marginal_1) - p_accident * integral_marginal_2

    def compute_dG_dphi2(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float,
                         z_idx: int) -> float:
        """Compute derivative of G(θ) with respect to φ₂(z)."""
        dp_da = self.functions.dp_da(a, self.params)
        p_accident = 1 - self.functions.p(a, self.params)

        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)

        phi2_val = phi2_values[z_idx]
        du_accident = self.functions.du_dx(self.W - phi1 + phi2_val - self.s, self.params)
        f_z = z_probs[z_idx]
        df_da_z = df_da[z_idx]

        return - dp_da * (du_accident * f_z) + p_accident * du_accident * df_da_z

    def compute_dG_da(self, theta: float, a: float, phi1: float, phi2_values: np.ndarray, delta: float) -> float:
        """Compute derivative of G(θ) with respect to action a (second-order derivative)."""
        dp_da = self.functions.dp_da(a, self.params)
        d2p_da2 = self.functions.d2p_da2(a, self.params)
        d2e_da2 = self.functions.d2e_da2(a, theta, self.params)

        p_no_accident = self.functions.p(a, self.params)
        p_accident = 1 - p_no_accident

        state_space = self.functions.f(a, delta, self.params)
        z_values, z_probs = state_space.get_all_states()
        df_da = self.functions.df_da(a, delta, self.params)

        # Compute second derivatives if available, otherwise use zero
        d2f_da2 = self.functions.d2f_da2(a, delta, self.params)

        utility_no_accident = self.functions.u(self.W - phi1, self.params)

        integral_u_f = 0.0
        integral_u_df_da = 0.0
        integral_u_d2f_da2 = 0.0

        for i, z in enumerate(z_values):
            phi2_val = phi2_values[i]
            u_accident = self.functions.u(self.W - phi1 + phi2_val - self.s, self.params)

            integral_u_f += u_accident * z_probs[i]
            integral_u_df_da += u_accident * df_da[i]
            integral_u_d2f_da2 += u_accident * d2f_da2[i]

        term1 = d2p_da2 * (utility_no_accident - integral_u_f)
        term2 = -2 * dp_da * integral_u_df_da
        term3 = p_accident * integral_u_d2f_da2
        term4 = -d2e_da2

        return term1 + term2 + term3 + term4

    def compute_insurer_profit(self, insurer_id: int, phi1: float, a_schedule: np.ndarray, phi2_values: np.ndarray,
                               other_phi1: float, other_a_schedule: np.ndarray, other_phi2_values: np.ndarray) -> float:
        """
        Compute expected profit for insurer i according to the formula:

        π_i = N * ∫_Θ P_i(θ) * [φ₁^i - (1-p(a^i(θ))) * ∫ φ₂^i(z) * f(z|a^i(θ),δ^i) dz] * h(θ) dθ - c(δ^i)

        Args:
            insurer_id: 1 or 2 for the insurer
            phi1: Premium for this insurer
            a_schedule: Action schedule for this insurer (array of length n_theta)
            phi2_values: Indemnity values for this insurer (array of length n_states)
            other_phi1: Premium for the other insurer
            other_a_schedule: Action schedule for the other insurer
            other_phi2_values: Indemnity values for the other insurer
        """
        delta = self.delta1 if insurer_id == 1 else self.delta2

        # Compute monitoring cost
        monitoring_cost = self.functions.m(delta, self.params)

        # Initialize profit integral
        profit_integral = 0.0

        # Integrate over all risk types
        for t, theta in enumerate(self.theta_grid):
            h_theta = self.h_theta[t]
            a_val = a_schedule[t]
            a_other = other_a_schedule[t]

            # Get choice probability P_i(θ)
            P0, P1, P2 = self.compute_choice_probabilities(theta, a_val, phi1, phi2_values,
                                                           a_other, other_phi1, other_phi2_values)
            Pi = P1 if insurer_id == 1 else P2

            # Compute probability of accident
            p_no_accident = self.functions.p(a_val, self.params)
            p_accident = 1 - p_no_accident

            # Get state space and compute indemnity integral
            state_space = self.functions.f(a_val, delta, self.params)
            z_values, z_probs = state_space.get_all_states()

            indemnity_integral = 0.0
            for j, z in enumerate(z_values):
                indemnity_integral += phi2_values[j] * z_probs[j]

            # Compute profit term for this risk type
            profit_term = phi1 - p_accident * indemnity_integral

            # Add to integral
            profit_integral += h_theta * Pi * profit_term

        # Final profit calculation
        expected_profit = self.N * profit_integral - monitoring_cost

        return expected_profit

    def multistart_solve(self, solver_name='ipopt', n_starts=10, verbose=True, save_plots=True, 
                        logger=None, executable_path=None, seed=42, save_all_solutions=True, 
                        save_model=False, model_filename=None):
        """
        Solve the duopoly equilibrium using Pyomo's standard multistart solver.
        
        For Nash equilibrium problems, we find ALL feasible equilibrium solutions,
        not just a single "best" solution, since there is no meaningful objective
        to optimize in equilibrium problems.
        
        Args:
            solver_name: Name of the solver to use
            n_starts: Number of different starting points to try
            verbose: Whether to print detailed output
            save_plots: Whether to save plots to files
            logger: SimulationLogger instance for recording results
            executable_path: Optional path to solver executable
            seed: Random seed for reproducibility
            save_all_solutions: Whether to save all feasible solutions found
            save_model: Whether to save the model in .nl format (only for first solve)
            model_filename: Optional filename for the saved model (without extension)
            
        Returns:
            Tuple of (success, solutions) where solutions is a list of dictionaries
        """
        # Set random seed for reproducibility
        import random
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"Starting Pyomo multistart optimization with {n_starts} different starting points...")

        if logger:
            logger.log_experiment_start("Multistart duopoly insurance simulation")
            logger.log_simulation_settings({
                'solver_name': solver_name,
                'n_starts': n_starts,
                'save_plots': save_plots,
                'save_all_solutions': save_all_solutions,
                'verbose': verbose,
                'executable_path': executable_path,
                'seed': seed,
                'save_model': save_model,
                'model_filename': model_filename
            })
            logger.log_parameters(self.params)

        try:
            # Build the model for multistart optimization
            model = self.build_model()
            
            # Configure the underlying solver with executable path if provided
            import os
            original_path = None
            if executable_path and solver_name == 'knitroampl':
                # Add the directory containing the executable to PATH
                exec_dir = os.path.dirname(executable_path)
                original_path = os.environ.get('PATH', '')
                os.environ['PATH'] = f"{exec_dir}:{original_path}"
                print(f"Added {exec_dir} to PATH for KNITRO executable")
            
            # Configure multistart solver
            multistart_solver = pyo.SolverFactory('multistart')
            
            # Set multistart options
            multistart_options = {
                'strategy': 'rand',  # Random choice between variable bounds
                'solver': solver_name,
                'solver_args': {},
                'iterations': n_starts,
                'stopping_mass': 0.5,
                'stopping_delta': 0.5,
                'suppress_unbounded_warning': True,
                'HCS_max_iterations': 1000,
                'HCS_tolerance': 0
            }
            
            print(f"Running multistart solver with {n_starts} iterations...")
            
            # Solve using multistart
            results = multistart_solver.solve(model, **multistart_options)
            
            # Check if multistart was successful
            if results.solver.status == SolverStatus.ok:
                print("✅ Pyomo multistart optimization completed successfully!")
                
                # Extract solution
                solution = self._extract_solution_from_model(model)
                
                if solution is not None:
                    # Store solution with metadata
                    solution_data = {
                        'solution': solution,
                        'timestamp': datetime.now().isoformat(),
                        'multistart_results': results,
                        'n_iterations': n_starts
                    }
                    
                    all_solutions = [solution_data]
                    
                    print(f"Number of equilibrium solutions found: {len(all_solutions)}")
                    
                    # Save all solutions if requested
                    if save_all_solutions and all_solutions:
                        self._save_all_solutions(all_solutions, logger)
                    
                    # Log final results
                    if logger:
                        logger.log_performance_metric("multistart_success_rate", 1.0)
                        logger.log_performance_metric("successful_solves", 1)
                        logger.log_performance_metric("total_solutions_found", len(all_solutions))
                    
                    # Save plots for solution analysis
                    if save_plots and all_solutions:
                        from utils.plotting import plot_results
                        
                        plots_dir = Path("plots")
                        plots_dir.mkdir(exist_ok=True)
                        
                        # Plot the solution
                        plot_path = plots_dir / "multistart_solution_1.png"
                        plot_results(self, solution_data['solution'], save_path=str(plot_path))
                        print(f"📈 Solution plot saved to: {plot_path}")
                        
                        if logger:
                            logger.log_plot_generation("multistart_solution_1", str(plot_path), "multistart_optimization")
                    
                    if logger:
                        logger.log_experiment_end("Pyomo multistart optimization completed successfully")
                        logger.print_summary()
                    
                    return True, all_solutions
                else:
                    print("❌ No valid solution extracted from multistart results")
                    return False, []
            else:
                print(f"❌ Pyomo multistart optimization failed with status: {results.solver.status}")
                return False, []
                
        except Exception as e:
            print(f"❌ Pyomo multistart optimization failed with error: {e}")
            if logger:
                logger.log_error(f"Pyomo multistart failed: {e}")
                logger.log_experiment_end("Pyomo multistart optimization failed")
            return False, []
    
    def _extract_solution_from_model(self, model):
        """
        Extract solution from the solved Pyomo model.
        
        Args:
            model: Solved Pyomo model
            
        Returns:
            Dictionary containing the solution or None if extraction fails
        """
        try:
            # Check if the model has been solved
            if not hasattr(model, 'phi1') or not hasattr(model, 'a') or not hasattr(model, 'phi2'):
                print("Error: Model does not have required variables")
                return None
            
            # Extract solution with proper error handling
            solution = {
                'insurer1': {
                    'phi1': float(pyo.value(model.phi1[1])) if model.phi1[1].value is not None else 0.0,
                    'a_schedule': np.array([float(pyo.value(model.a[1, t])) if model.a[1, t].value is not None else 0.0 for t in model.THETA]),
                    'phi2_values': np.array([float(pyo.value(model.phi2[1, z])) if model.phi2[1, z].value is not None else 0.0 for z in model.Z])
                },
                'insurer2': {
                    'phi1': float(pyo.value(model.phi1[2])) if model.phi1[2].value is not None else 0.0,
                    'a_schedule': np.array([float(pyo.value(model.a[2, t])) if model.a[2, t].value is not None else 0.0 for t in model.THETA]),
                    'phi2_values': np.array([float(pyo.value(model.phi2[2, z])) if model.phi2[2, z].value is not None else 0.0 for z in model.Z])
                },
                'lagrange_multipliers': {
                    'insurer1': np.array([float(pyo.value(model.lam[1, t])) if model.lam[1, t].value is not None else 0.0 for t in model.THETA]),
                    'insurer2': np.array([float(pyo.value(model.lam[2, t])) if model.lam[2, t].value is not None else 0.0 for t in model.THETA])
                }
            }
            
            print(f"Successfully extracted solution:")
            print(f"  Insurer 1 premium: {solution['insurer1']['phi1']:.4f}")
            print(f"  Insurer 2 premium: {solution['insurer2']['phi1']:.4f}")
            
            return solution
        except Exception as e:
            print(f"Error extracting solution: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_all_solutions(self, all_solutions, logger=None):
        """
        Save all feasible equilibrium solutions found during multistart to files.
        
        Args:
            all_solutions: List of solution dictionaries with metadata
            logger: Optional logger for recording
        """
        
        # Create solutions directory
        solutions_dir = Path("multistart_solutions")
        solutions_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed solutions
        solutions_file = solutions_dir / f"all_solutions_{timestamp}.json"
        
        # Prepare solutions for JSON serialization
        import json
        serializable_solutions = []
        
        for i, solution_data in enumerate(all_solutions):
            solution = solution_data['solution']
            
            # Calculate expected profits for both insurers
            insurer1_profit = self.compute_insurer_profit(
                insurer_id=1,
                phi1=solution['insurer1']['phi1'],
                a_schedule=np.array(solution['insurer1']['a_schedule']),
                phi2_values=np.array(solution['insurer1']['phi2_values']),
                other_phi1=solution['insurer2']['phi1'],
                other_a_schedule=np.array(solution['insurer2']['a_schedule']),
                other_phi2_values=np.array(solution['insurer2']['phi2_values'])
            )
            
            insurer2_profit = self.compute_insurer_profit(
                insurer_id=2,
                phi1=solution['insurer2']['phi1'],
                a_schedule=np.array(solution['insurer2']['a_schedule']),
                phi2_values=np.array(solution['insurer2']['phi2_values']),
                other_phi1=solution['insurer1']['phi1'],
                other_a_schedule=np.array(solution['insurer1']['a_schedule']),
                other_phi2_values=np.array(solution['insurer1']['phi2_values'])
            )
            
            # Create serializable solution with profit information
            serializable_solution = {
                'solution_id': i + 1,
                'insurer1': {
                    'phi1': float(solution['insurer1']['phi1']),
                    'phi2_values': [float(x) for x in solution['insurer1']['phi2_values']],
                    'a_schedule': [float(x) for x in solution['insurer1']['a_schedule']],
                    'expected_profit': float(insurer1_profit),
                    'lagrange_multipliers': [float(x) for x in solution['lagrange_multipliers']['insurer1']],
                },
                'insurer2': {
                    'phi1': float(solution['insurer2']['phi1']),
                    'phi2_values': [float(x) for x in solution['insurer2']['phi2_values']],
                    'a_schedule': [float(x) for x in solution['insurer2']['a_schedule']],
                    'expected_profit': float(insurer2_profit),
                    'lagrange_multipliers': [float(x) for x in solution['lagrange_multipliers']['insurer2']],
                },
                'n_iterations': solution_data.get('n_iterations', 0)
            }
            
            serializable_solutions.append(serializable_solution)
        
        # Save to JSON file
        with open(solutions_file, 'w') as f:
            json.dump(serializable_solutions, f, indent=2)
        
        print(f"Saved {len(serializable_solutions)} equilibrium solutions to {solutions_file}")
        
        # Log file generation if logger provided
        if logger:
            logger.log_file_generation("all_solutions", str(solutions_file), "multistart_optimization")
