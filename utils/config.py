"""
Utility Functions Module

Contains utility functions for solver configuration and other helper functions.
"""

def get_solver_options(solver_name: str, verbose: bool = False, debug_mode: bool = False) -> dict:
    """
    Get solver-specific options for different solvers.
    
    Args:
        solver_name: Name of the solver
        verbose: Whether to print detailed output
        debug_mode: Whether to enable debug mode
        
    Returns:
        Dictionary of solver options
    """
    if solver_name == 'ipopt':
        base_options = {
            'max_iter': 5000,
            'tol': 1e-4,
            'print_level': 5 if verbose or debug_mode else 0,
            'output_file': 'ipopt_debug.out' if debug_mode else None,
            'linear_solver': 'mumps',
            'hessian_approximation': 'limited-memory',
            'mu_strategy': 'adaptive',
            'bound_push': 1e-8,
            'bound_frac': 1e-8,
            'dual_inf_tol': 1e-4,
            'compl_inf_tol': 1e-4,
            'acceptable_tol': 1e-3,
            'acceptable_iter': 10,
            'mehrotra_algorithm': 'yes',
            'warm_start_init_point': 'yes',
            'nlp_scaling_method': 'gradient-based'
        }
        
        return base_options
        
    elif solver_name == 'knitroampl':
        base_options = {
            'outlev': 3 if verbose or debug_mode else 0,
            'maxit': 5000,
            'feastol': 1e-4,
            'opttol': 1e-4,
            'honorbnds': 1,
            'bar_murule': 4,
            'bar_initpt': 3,
            'linsolver': 1,
            'hessopt': 2,
            'presolve': 1
        }
        
        return base_options
        
    elif solver_name == 'conopt':
        base_options = {
            'limrow': 0,
            'limcol': 0,
            'iterlim': 5000,
            'reslim': 1000,
            'optca': 1e-4,
            'optcr': 1e-4,
            'lrf': 0.1,
            'lrs': 0.1,
            'bratio': 0.1
        }
        
        return base_options
        
    elif solver_name == 'baron':
        base_options = {
            'maxtime': 3600,
            'maxiter': 10000,
            'outlev': 3 if verbose or debug_mode else 0,
            'epsr': 1e-4,
            'epsa': 1e-4,
            'epso': 1e-4,
            'epsint': 1e-6,
            'epscut': 1e-6,
            'prfreq': 100
        }
        
        return base_options
        
    elif solver_name == 'scip':
        base_options = {
            'display/verblevel': 4 if verbose or debug_mode else 0,
            'limits/time': 3600,
            'limits/iterations': 10000,
            'limits/nodes': 100000,
            'limits/gap': 1e-4,
            'numerics/feastol': 1e-6,
            'lp/initalgorithm': 'd',
            'lp/resolvealgorithm': 'd'
        }
        
        return base_options
        
    else:
        # Default options for other solvers
        base_options = {
            'print_level': 3 if verbose or debug_mode else 0,
            'max_iter': 5000,
            'tol': 1e-4
        }
        
        return base_options 