"""
Solver Methods Module

Contains the large methods for the DuopolySolver class that are separated
to keep the main helper.py file manageable.
"""

import numpy as np
import pyomo.environ as pyo
from pathlib import Path
import time
from utils.config import get_solver_options
from utils.plotting import plot_results

# Pyomo imports for KKT-based solving
try:
    from pyomo.opt import SolverStatus, TerminationCondition
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False

# Pyomo MPEC imports for complementarity constraints
try:
    from pyomo.mpec import complements, Complementarity
    MPEC_AVAILABLE = True
except ImportError:
    MPEC_AVAILABLE = False
    print("Warning: Pyomo MPEC module not available. Using relaxed complementary slackness constraints.")


def build_and_solve_model(self, solver_name='ipopt', verbose=False, debug_mode=True, executable_path=None):
    """
    Solve the duopoly equilibrium using KKT conditions with pyomo.

    """

    if not PYOMO_AVAILABLE:
        raise ImportError("Pyomo is required for KKT-based solving. Please install it with: pip install pyomo")
    
    print("Setting up KKT system for duopoly equilibrium...")
    
    # Create the optimization model
    model = pyo.ConcreteModel()
    
    # Sets
    model.THETA = pyo.Set(initialize=range(self.n_theta))  # Risk types
    model.Z = pyo.Set(initialize=range(self.n_states))     # States
    model.I = pyo.Set(initialize=[1, 2])                  # Insurers
    
    # Parameters
    model.W = pyo.Param(initialize=self.W)
    model.s = pyo.Param(initialize=self.s)
    model.N = pyo.Param(initialize=self.N)
    model.theta_vals = pyo.Param(model.THETA, initialize={i: self.theta_grid[i] for i in range(self.n_theta)})
    model.h_theta = pyo.Param(model.THETA, initialize={i: self.h_theta[i] for i in range(self.n_theta)})
    model.z_vals = pyo.Param(model.Z, initialize={i: self.z_values[i] for i in range(self.n_states)})
    model.delta1 = pyo.Param(initialize=self.delta1)
    model.delta2 = pyo.Param(initialize=self.delta2)
    
    # Decision Variables for both insurers
    model.a = pyo.Var(model.I, model.THETA, domain=pyo.NonNegativeReals, 
                     bounds=(self.a_min, self.a_max))
    model.phi1 = pyo.Var(model.I, domain=pyo.NonNegativeReals, bounds=(0, self.s))
    model.phi2 = pyo.Var(model.I, model.Z, domain=pyo.NonNegativeReals, bounds=(0, self.s))
    
    # Lagrange multipliers
    model.lam = pyo.Var(model.I, model.THETA, domain=pyo.Reals)
    model.nu_L = pyo.Var(model.I, model.THETA, domain=pyo.NonNegativeReals)
    model.nu_U = pyo.Var(model.I, model.THETA, domain=pyo.NonNegativeReals)
    model.eta = pyo.Var(model.I, domain=pyo.NonNegativeReals)
    model.gamma = pyo.Var(model.I, model.Z, domain=pyo.NonNegativeReals)
    
    # Set starting values if provided
    if hasattr(self, 'starting_point') and self.starting_point is not None:
        print("Setting starting values from multistart...")
        start_point = self.starting_point
        
        # Set starting values for action levels
        for i in model.I:
            for t in model.THETA:
                if (i, t) in start_point['a']:
                    model.a[i, t].value = start_point['a'][(i, t)]
        
        # Set starting values for premiums
        for i in model.I:
            if i in start_point['phi1']:
                model.phi1[i].value = start_point['phi1'][i]
        
        # Set starting values for indemnities
        for i in model.I:
            for z in model.Z:
                if (i, z) in start_point['phi2']:
                    model.phi2[i, z].value = start_point['phi2'][(i, z)]
        
        # Set starting values for Lagrange multipliers
        for i in model.I:
            for t in model.THETA:
                if (i, t) in start_point['lam']:
                    model.lam[i, t].value = start_point['lam'][(i, t)]
                if (i, t) in start_point['nu_L']:
                    model.nu_L[i, t].value = start_point['nu_L'][(i, t)]
                if (i, t) in start_point['nu_U']:
                    model.nu_U[i, t].value = start_point['nu_U'][(i, t)]
            
            if i in start_point['eta']:
                model.eta[i].value = start_point['eta'][i]
            
            for z in model.Z:
                if (i, z) in start_point['gamma']:
                    model.gamma[i, z].value = start_point['gamma'][(i, z)]
    
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
    
    # Complementary slackness constraints using MPEC formulation
    if MPEC_AVAILABLE:
        print("Using MPEC complementarity constraints...")
        
        # Action lower bound complementarity: ν_L^i(θ) ⟂ (a^i(θ) - a_min)
        def action_lower_bound_rule(model, i, t):
            return complements(model.nu_L[i, t] >= 0, model.a[i, t] - self.a_min >= 0)
        
        model.action_lower_bound = Complementarity(model.I, model.THETA, rule=action_lower_bound_rule)
        
        # Action upper bound complementarity: ν_U^i(θ) ⟂ (a_max - a^i(θ))
        def action_upper_bound_rule(model, i, t):
            return complements(model.nu_U[i, t] >= 0, self.a_max - model.a[i, t] >= 0)
        
        model.action_upper_bound = Complementarity(model.I, model.THETA, rule=action_upper_bound_rule)
        
        # Premium non-negativity complementarity: η^i ⊥ φ₁^i
        def premium_nonnegativity_rule(model, i):
            return complements(model.eta[i] >= 0, model.phi1[i] >= 0)
        
        model.premium_nonnegativity = Complementarity(model.I, rule=premium_nonnegativity_rule)
        
        # Indemnity non-negativity complementarity: γ^i(z) ⊥ φ₂^i(z)
        def indemnity_nonnegativity_rule(model, i, z):
            return complements(model.gamma[i, z] >= 0, model.phi2[i, z] >= 0)
        
        model.indemnity_nonnegativity = Complementarity(model.I, model.Z, rule=indemnity_nonnegativity_rule)
        
    else:
        print("MPEC not available, using relaxed complementary slackness constraints...")
        
        # Fallback to relaxed complementary slackness constraints
        def comp_slack_lower_rule(model, i, t):
            return model.nu_L[i, t] * (self.a_min - model.a[i, t]) == 0
        
        model.comp_slack_lower = pyo.Constraint(model.I, model.THETA, rule=comp_slack_lower_rule)
        
        def comp_slack_upper_rule(model, i, t):
            return model.nu_U[i, t] * (model.a[i, t] - self.a_max) == 0
        
        model.comp_slack_upper = pyo.Constraint(model.I, model.THETA, rule=comp_slack_upper_rule)
        
        def comp_slack_premium_rule(model, i):
            return model.eta[i] * model.phi1[i] == 0
        
        model.comp_slack_premium = pyo.Constraint(model.I, rule=comp_slack_premium_rule)
        
        def comp_slack_indemnity_rule(model, i, z):
            return model.gamma[i, z] * model.phi2[i, z] == 0
        
        model.comp_slack_indemnity = pyo.Constraint(model.I, model.Z, rule=comp_slack_indemnity_rule)
    
    # Stationarity conditions
    def stationarity_action_rule(model, i, t):
        """Stationarity with respect to action a^i(θ)."""
        theta = model.theta_vals[t]
        a_val = model.a[i, t]
        phi1_val = model.phi1[i]
        
        # Get monitoring level
        delta_val = model.delta1 if i == 1 else model.delta2
        
        # Get phi2 values for all states
        phi2_values = np.array([model.phi2[i, z] for z in model.Z])
        
        # Get contract values for the other insurer
        other_insurer = 2 if i == 1 else 1
        a_other = model.a[other_insurer, t]
        phi1_other = model.phi1[other_insurer]
        phi2_other_values = np.array([model.phi2[other_insurer, z] for z in model.Z])
        
        # Use existing helper functions to compute choice probabilities
        P0, P1, P2 = self.compute_choice_probabilities(theta, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
        Pi = P1 if i == 1 else P2
        
        # Use existing helper function to compute ∂P_i(θ)/∂a^i(θ)
        dPi_da = self.compute_dPi_da(theta, i, a_val, phi1_val, phi2_values, a_other, phi1_other, phi2_other_values)
        
        # Use existing helper function to compute ∂G(θ)/∂a^i(θ)
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
        
        # KKT stationarity condition according to mathematical model
        term1 = self.N * dPi_da * profit_term
        term2 = -self.N * Pi * second_term_bracket
        term3 = model.lam[i, t] * dG_da
        
        return term1 + term2 + term3 - model.nu_L[i, t] + model.nu_U[i, t] == 0
    
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
        return self.N * integral_term + lambda_dG_integral - model.eta[i] == 0
    
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
        return -self.N * first_integral + self.N * second_integral + lambda_dG_integral - model.gamma[i, z] == 0
    
    model.stationarity_indemnity = pyo.Constraint(model.I, model.Z, rule=stationarity_indemnity_rule)
    
    # Objective function - 0
    model.obj = pyo.Objective(expr=0, sense=pyo.minimize)
    
    # Debug mode: Analyze model structure
    if debug_mode:
        print("\n" + "="*50)
        print("DEBUG MODE: MODEL ANALYSIS")
        print("="*50)
        
        # Count variables and constraints
        total_vars = len(model.component_map(pyo.Var))
        total_constrs = len(model.component_map(pyo.Constraint))
        # Count MPEC complementarity constraints if available
        total_mpec_constrs = 0
        
        print(f"Model Statistics:")
        print(f"  - Variables: {total_vars}")
        print(f"  - Regular Constraints: {total_constrs}")
        print(f"  - MPEC Constraints: {total_mpec_constrs}")
        print(f"  - Total Constraints: {total_constrs + total_mpec_constrs}")
        print(f"  - C/V Ratio: {(total_constrs + total_mpec_constrs)/total_vars:.2f}")

    # Solve the model
    print(f"\nSolving KKT system with {solver_name} solver...")
    try:
        # Create solver with optional executable path
        if executable_path and solver_name == 'knitroampl':
            print(f"Using KNITRO executable: {executable_path}")
            opt = pyo.SolverFactory(solver_name, executable=executable_path)
        else:
            opt = pyo.SolverFactory(solver_name)
        
        # Use centralized solver options for all solvers
        # Check if we're using MPEC formulation
        is_mpec_problem = MPEC_AVAILABLE and hasattr(model, 'action_lower_bound')
        solver_options = get_solver_options(solver_name, verbose=verbose, debug_mode=debug_mode)
        
        if is_mpec_problem:
            print("Using MPEC-specific solver options for complementarity constraints...")
        
        if solver_options:
            opt.options.update({k: v for k, v in solver_options.items() if v is not None})
        
        start_time = time.time()
        results = opt.solve(model, tee=verbose or debug_mode)
        solve_time = time.time() - start_time
        
        # Enhanced result analysis
        if debug_mode:
            print(f"\n" + "="*50)
            print("DEBUG MODE: SOLVER RESULTS ANALYSIS")
            print("="*50)
            print(f"Solver Status: {results.solver.status}")
            print(f"Termination Condition: {results.solver.termination_condition}")
            print(f"Solve Time: {solve_time:.3f} seconds")
            
            if hasattr(results.solver, 'iterations'):
                print(f"Iterations: {results.solver.iterations}")
            
            if hasattr(results.solver, 'objective_value'):
                print(f"Objective Value: {results.solver.objective_value}")
            
            # Read detailed solver output if available
            try:
                with open('ipopt_debug.out', 'r') as f:
                    solver_output = f.read()
                    print(f"\nDetailed Solver Output (last 20 lines):")
                    lines = solver_output.strip().split('\n')
                    for line in lines[-20:]:
                        print(f"  {line}")
            except FileNotFoundError:
                print("No detailed solver output file found.")
        
        if (results.solver.status == SolverStatus.ok and 
            results.solver.termination_condition == TerminationCondition.optimal):
            
            print("KKT system solved successfully!")
            
            # Extract solution
            solution = {
                'insurer1': {
                    'phi1': pyo.value(model.phi1[1]),
                    'phi2': np.array([pyo.value(model.phi2[1, z]) for z in model.Z]),
                    'a_schedule': np.array([pyo.value(model.a[1, t]) for t in model.THETA]),
                    'multipliers': {
                        'lambda': np.array([pyo.value(model.lam[1, t]) for t in model.THETA]),
                        'nu_L': np.array([pyo.value(model.nu_L[1, t]) for t in model.THETA]),
                        'nu_U': np.array([pyo.value(model.nu_U[1, t]) for t in model.THETA]),
                        'eta': pyo.value(model.eta[1]),
                        'gamma': np.array([pyo.value(model.gamma[1, z]) for z in model.Z])
                    }
                },
                'insurer2': {
                    'phi1': pyo.value(model.phi1[2]),
                    'phi2': np.array([pyo.value(model.phi2[2, z]) for z in model.Z]),
                    'a_schedule': np.array([pyo.value(model.a[2, t]) for t in model.THETA]),
                    'multipliers': {
                        'lambda': np.array([pyo.value(model.lam[2, t]) for t in model.THETA]),
                        'nu_L': np.array([pyo.value(model.nu_L[2, t]) for t in model.THETA]),
                        'nu_U': np.array([pyo.value(model.nu_U[2, t]) for t in model.THETA]),
                        'eta': pyo.value(model.eta[2]),
                        'gamma': np.array([pyo.value(model.gamma[2, z]) for z in model.Z])
                    }
                },
                'solver_status': 'optimal',
                'solver_info': results,
                'solve_time': solve_time
            }
            
            return solution
            
        else:
            print(f"\nSolver failed!")
            print(f"  Status: {results.solver.status}")
            print(f"  Termination: {results.solver.termination_condition}")
            
            # Provide specific guidance based on termination condition
            if results.solver.termination_condition == TerminationCondition.infeasible:
                print(f"  REASON: Problem is infeasible")
            elif results.solver.termination_condition == TerminationCondition.infeasibleOrUnbounded:
                print(f"  REASON: Problem is infeasible or unbounded")
            elif results.solver.termination_condition == TerminationCondition.maxIterations:
                print(f"  REASON: Maximum iterations reached")
            elif results.solver.termination_condition == TerminationCondition.other:
                print(f"  REASON: Other termination condition")
            
            return None
            
    except Exception as e:
        print(f"Error solving KKT system: {e}")
        if debug_mode:
            import traceback
            print("Full traceback:")
            traceback.print_exc()
        return None


def run(self, solver_name='ipopt', verbose=True, save_plots=True, logger=None, executable_path=None):
    """
    Run KKT-based simulation for duopoly insurance model.
    
    Args:
        solver_name: Name of the solver to use
        verbose: Whether to print detailed output
        save_plots: Whether to save plots to files
        logger: SimulationLogger instance for recording results
        executable_path: Optional path to solver executable (useful for KNITRO)
        
    Returns:
        Tuple of (success, solution) where solution is a dictionary containing simulation results
    """
    print("Testing KKT solver...")
    print(f"Model configuration:")
    print(f"  - Risk types: {self.n_theta}")
    print(f"  - States: {self.n_states}")
    print(f"  - Action bounds: [{self.a_min}, {self.a_max}]")
    print(f"  - Initial wealth W: {self.W}")
    print(f"  - Accident severity s: {self.s}")
    print(f"  - Monitoring levels: δ₁={self.delta1}, δ₂={self.delta2}")

    # Log experiment start if logger is provided
    if logger:
        logger.log_experiment_start("Duopoly insurance simulation")
        logger.log_simulation_settings({
            'solver_name': solver_name,
            'save_plots': save_plots,
            'verbose': verbose,
            'executable_path': executable_path
        })
        logger.log_parameters(self.params)

    try:
        start_time = time.time()
        solution = build_and_solve_model(self, solver_name=solver_name, verbose=verbose, executable_path=executable_path)
        solve_time = time.time() - start_time

        if solution is not None:
            print("\n" + "="*50)
            print("KKT SOLVER SUCCESS!")
            print("="*50)

            for insurer_id in [1, 2]:
                key = f'insurer{insurer_id}'
                print(f"\nInsurer {insurer_id} Solution:")
                print(f"  Premium φ₁^{insurer_id}: {solution[key]['phi1']:.4f}")
                print(f"  Indemnities φ₂^{insurer_id}: {solution[key]['phi2']}")
                print(f"  Action schedule a^{insurer_id}(θ): {solution[key]['a_schedule']}")

            # Log performance metrics if logger is provided
            if logger:
                logger.log_performance_metric("solve_time_seconds", solve_time)
                
                # Log the solution
                logger.log_solution_summary({
                    'insurer1_premium': solution['insurer1']['phi1'],
                    'insurer2_premium': solution['insurer2']['phi1'],
                    'solve_time': solve_time
                })

            # Save plots if requested
            if save_plots:
                plots_dir = Path("plots")
                plots_dir.mkdir(exist_ok=True)
                
                # Comprehensive results plot
                plot_path = plots_dir / "comprehensive_results.png"
                plot_results(self, solution, save_path=str(plot_path))
                
                if logger:
                    logger.log_plot_generation("comprehensive_results", str(plot_path), "single_configuration")

            # Log experiment end if logger is provided
            if logger:
                logger.log_experiment_end("Simulation completed successfully")
                logger.print_summary()

            return True, solution
        else:
            print("KKT solver failed to find a solution.")
            
            if logger:
                logger.log_error("Solver failed to converge")
                logger.log_experiment_end("Simulation failed")
            
            return False, None

    except Exception as e:
        error_msg = f"Error testing KKT solver: {e}"
        print(error_msg)
        
        if logger:
            logger.log_error(error_msg)
            logger.log_experiment_end("Simulation failed with error")
        
        return False, None
