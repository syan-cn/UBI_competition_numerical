"""
Simulation Logger Module

A comprehensive logging system for recording insurance simulation settings,
parameters, results, and analysis in a structured and searchable format.
"""

import json
import logging
import os
import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path


class SimulationLogger:
    """
    A comprehensive logger for insurance simulation experiments.
    
    This logger can record:
    - Simulation settings and parameters
    - Function configurations
    - Optimization results
    - Sensitivity analysis results
    - Performance metrics
    - Error logs and warnings
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 experiment_name: Optional[str] = None,
                 log_level: str = "INFO",
                 save_to_file: bool = True,
                 save_to_json: bool = True,
                 save_to_csv: bool = True):
        """
        Initialize the simulation logger.
        
        Args:
            log_dir: Directory to store log files
            experiment_name: Name for this experiment (auto-generated if None)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            save_to_file: Whether to save logs to text files
            save_to_json: Whether to save structured data to JSON
            save_to_csv: Whether to save tabular data to CSV
        """
        # Base directory for all experiments
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # ------------------------------------------------------------------
        # Determine a unique experiment name and dedicated sub-folder so that
        # data from different runs are never overwritten. If the user passes
        # a name that has already been used, we append a timestamp suffix.
        # ------------------------------------------------------------------
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            self.experiment_name = f"experiment_{timestamp}"
        else:
            tentative_name = experiment_name
            # If a folder (or file set) with the same name already exists, add a timestamp
            existing_path = self.log_dir / tentative_name
            if existing_path.exists():
                tentative_name = f"{tentative_name}_{timestamp}"
            self.experiment_name = tentative_name

        # Create a dedicated directory for this experiment
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_to_file = save_to_file
        self.save_to_json = save_to_json
        self.save_to_csv = save_to_csv
        
        # Initialize logging
        self._setup_logging(log_level)
        
        # Data storage
        self.simulation_data = {
            'experiment_info': {
                'name': self.experiment_name,
                'start_time': datetime.datetime.now().isoformat(),
                'version': '1.0'
            },
            'settings': {},
            'parameters': {},
            'function_configs': {},
            'results': {},
            'sensitivity_analysis': {},
            'performance_metrics': {},
            'errors': [],
            'warnings': []
        }
        
        self.logger.info(f"Simulation logger initialized for experiment: {self.experiment_name}")
    
    def _setup_logging(self, log_level: str):
        """Setup the logging configuration."""
        self.logger = logging.getLogger(f"simulation_logger_{self.experiment_name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.save_to_file:
            log_file = self.experiment_dir / f"{self.experiment_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_experiment_start(self, description: str = ""):
        """Log the start of an experiment."""
        self.simulation_data['experiment_info']['description'] = description
        self.simulation_data['experiment_info']['start_time'] = datetime.datetime.now().isoformat()
        
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        if description:
            self.logger.info(f"Description: {description}")
    
    def log_experiment_end(self, summary: str = ""):
        """Log the end of an experiment."""
        self.simulation_data['experiment_info']['end_time'] = datetime.datetime.now().isoformat()
        self.simulation_data['experiment_info']['summary'] = summary
        
        self.logger.info(f"Experiment completed: {self.experiment_name}")
        if summary:
            self.logger.info(f"Summary: {summary}")
        
        # Save all data
        self._save_all_data()
    
    def log_simulation_settings(self, settings: Dict[str, Any]):
        """Log simulation settings and configuration."""
        self.simulation_data['settings'] = settings.copy()
        
        self.logger.info("Logging simulation settings:")
        for key, value in settings.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log simulation parameters."""
        self.simulation_data['parameters'] = params.copy()
        
        self.logger.info("Logging simulation parameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_function_config(self, function_config: Dict[str, str], state_config: Optional[Dict] = None):
        """Log function configuration."""
        config_key = state_config['name'] if state_config else 'default'
        self.simulation_data['function_configs'][config_key] = {
            'functions': function_config.copy(),
            'state_config': state_config.copy() if state_config else None
        }
        
        self.logger.info(f"Logging function configuration for {config_key}:")
        for key, value in function_config.items():
            self.logger.info(f"  {key}: {value}")
        if state_config:
            self.logger.info(f"  State config: {state_config}")
    
    def log_optimization_start(self, insurer_id: int, state_name: str = "default"):
        """Log the start of optimization for an insurer."""
        self.logger.info(f"Starting optimization for Insurer {insurer_id} ({state_name})")
    
    def log_optimization_result(self, 
                              insurer_id: int, 
                              result: Dict[str, Any], 
                              state_name: str = "default",
                              optimization_info: Optional[Dict] = None):
        """Log optimization results for an insurer."""
        if 'results' not in self.simulation_data:
            self.simulation_data['results'] = {}
        
        if state_name not in self.simulation_data['results']:
            self.simulation_data['results'][state_name] = {}
        
        # Store the result
        self.simulation_data['results'][state_name][f'insurer_{insurer_id}'] = {
            'premium': float(result.get('premium', 0)),
            'indemnity_values': result.get('indemnity_values', []).tolist() if isinstance(result.get('indemnity_values'), np.ndarray) else result.get('indemnity_values', []),
            'action_schedule': result.get('a_schedule', []).tolist() if isinstance(result.get('a_schedule'), np.ndarray) else result.get('a_schedule', []),
            'expected_profit': float(result.get('expected_profit', 0)),
            'optimization_info': optimization_info or {}
        }
        
        self.logger.info(f"Insurer {insurer_id} optimization completed:")
        self.logger.info(f"  Premium: {result.get('premium', 0):.2f}")
        self.logger.info(f"  Expected Profit: {result.get('expected_profit', 0):.2f}")
        self.logger.info(f"  Average Action: {np.mean(result.get('a_schedule', [0])):.4f}")
        self.logger.info(f"  Average Indemnity: {np.mean(result.get('indemnity_values', [0])):.2f}")
    
    def log_duopoly_solution(self, solution: Dict[str, Any], state_name: str = "default"):
        """Log the complete duopoly solution."""
        if 'results' not in self.simulation_data:
            self.simulation_data['results'] = {}
        
        if state_name not in self.simulation_data['results']:
            self.simulation_data['results'][state_name] = {}
        
        # Store complete solution
        self.simulation_data['results'][state_name]['duopoly_solution'] = {
            'insurer1': {
                'premium': float(solution['insurer1']['premium']),
                'indemnity_values': solution['insurer1']['indemnity_values'].tolist() if isinstance(solution['insurer1']['indemnity_values'], np.ndarray) else solution['insurer1']['indemnity_values'],
                'action_schedule': solution['insurer1']['a_schedule'].tolist() if isinstance(solution['insurer1']['a_schedule'], np.ndarray) else solution['insurer1']['a_schedule'],
                'expected_profit': float(solution['insurer1']['expected_profit'])
            },
            'insurer2': {
                'premium': float(solution['insurer2']['premium']),
                'indemnity_values': solution['insurer2']['indemnity_values'].tolist() if isinstance(solution['insurer2']['indemnity_values'], np.ndarray) else solution['insurer2']['indemnity_values'],
                'action_schedule': solution['insurer2']['a_schedule'].tolist() if isinstance(solution['insurer2']['a_schedule'], np.ndarray) else solution['insurer2']['a_schedule'],
                'expected_profit': float(solution['insurer2']['expected_profit'])
            },
            'market_share': solution.get('market_share', {}),
            'total_profit': float(solution.get('total_profit', 0))
        }
        
        self.logger.info(f"Duopoly solution completed for {state_name}:")
        self.logger.info(f"  Insurer 1 - Premium: {solution['insurer1']['premium']:.2f}, Profit: {solution['insurer1']['expected_profit']:.2f}")
        self.logger.info(f"  Insurer 1 - Action Schedule: {np.array2string(solution['insurer1']['a_schedule'], precision=4, separator=', ')}")
        self.logger.info(f"  Insurer 1 - Indemnity Values: {np.array2string(solution['insurer1']['indemnity_values'], precision=2, separator=', ')}")
        
        self.logger.info(f"  Insurer 2 - Premium: {solution['insurer2']['premium']:.2f}, Profit: {solution['insurer2']['expected_profit']:.2f}")
        self.logger.info(f"  Insurer 2 - Action Schedule: {np.array2string(solution['insurer2']['a_schedule'], precision=4, separator=', ')}")
        self.logger.info(f"  Insurer 2 - Indemnity Values: {np.array2string(solution['insurer2']['indemnity_values'], precision=2, separator=', ')}")
        
        self.logger.info(f"  Total Market Profit: {solution.get('total_profit', 0):.2f}")
    
    def log_sensitivity_analysis(self, 
                               param_name: str, 
                               param_range: List[float], 
                               results: List[Dict[str, Any]],
                               state_name: str = "default"):
        """Log sensitivity analysis results."""
        if 'sensitivity_analysis' not in self.simulation_data:
            self.simulation_data['sensitivity_analysis'] = {}
        
        if state_name not in self.simulation_data['sensitivity_analysis']:
            self.simulation_data['sensitivity_analysis'][state_name] = {}
        
        # Store sensitivity analysis data
        self.simulation_data['sensitivity_analysis'][state_name][param_name] = {
            'param_range': param_range,
            'results': results
        }
        
        self.logger.info(f"Sensitivity analysis completed for {param_name} ({state_name}):")
        self.logger.info(f"  Parameter range: {param_range}")
        self.logger.info(f"  Number of points: {len(results)}")
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics."""
        if 'performance_metrics' not in self.simulation_data:
            self.simulation_data['performance_metrics'] = {}
        
        self.simulation_data['performance_metrics'][metric_name] = {
            'value': value,
            'unit': unit,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Performance metric - {metric_name}: {value} {unit}")
    
    def log_error(self, error_message: str, error_type: str = "ERROR", details: Optional[Dict] = None):
        """Log errors."""
        error_entry = {
            'message': error_message,
            'type': error_type,
            'timestamp': datetime.datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.simulation_data['errors'].append(error_entry)
        self.logger.error(f"{error_type}: {error_message}")
        if details:
            self.logger.error(f"Details: {details}")
    
    def log_warning(self, warning_message: str, details: Optional[Dict] = None):
        """Log warnings."""
        warning_entry = {
            'message': warning_message,
            'timestamp': datetime.datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.simulation_data['warnings'].append(warning_entry)
        self.logger.warning(f"WARNING: {warning_message}")
        if details:
            self.logger.warning(f"Details: {details}")
    
    def log_plot_generation(self, plot_type: str, save_path: str, state_name: str = "default"):
        """Log plot generation."""
        self.logger.info(f"Generated {plot_type} plot for {state_name}: {save_path}")
    
    def _save_all_data(self):
        """Save all logged data to files."""
        if self.save_to_json:
            self._save_json_data()
        
        if self.save_to_csv:
            self._save_csv_data()
    
    def _save_json_data(self):
        """Save structured data to JSON file."""
        json_file = self.experiment_dir / f"{self.experiment_name}_data.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_data = convert_for_json(self.simulation_data)
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        self.logger.info(f"Data saved to JSON: {json_file}")
    
    def _save_csv_data(self):
        """Save tabular data to CSV files."""
        csv_dir = self.experiment_dir / "csv_data"
        csv_dir.mkdir(exist_ok=True)
        
        # Save results summary
        if 'results' in self.simulation_data:
            results_data = []
            for state_name, state_results in self.simulation_data['results'].items():
                if 'duopoly_solution' in state_results:
                    solution = state_results['duopoly_solution']
                    
                    # Basic info
                    row_data = {
                        'state_name': state_name,
                        'insurer1_premium': solution['insurer1']['premium'],
                        'insurer1_profit': solution['insurer1']['expected_profit'],
                        'insurer2_premium': solution['insurer2']['premium'],
                        'insurer2_profit': solution['insurer2']['expected_profit'],
                        'total_profit': solution.get('total_profit', 0)
                    }
                    
                    # Add all action schedules
                    for i, action in enumerate(solution['insurer1']['action_schedule']):
                        row_data[f'insurer1_action_theta{i+1}'] = action
                    for i, action in enumerate(solution['insurer2']['action_schedule']):
                        row_data[f'insurer2_action_theta{i+1}'] = action
                        
                    # Add all indemnity values
                    for i, indemnity in enumerate(solution['insurer1']['indemnity_values']):
                        row_data[f'insurer1_indemnity_z{i+1}'] = indemnity
                    for i, indemnity in enumerate(solution['insurer2']['indemnity_values']):
                        row_data[f'insurer2_indemnity_z{i+1}'] = indemnity
                        
                    results_data.append(row_data)
            
            if results_data:
                df_results = pd.DataFrame(results_data)
                df_results.to_csv(csv_dir / f"{self.experiment_name}_results.csv", index=False)
        
        # Save sensitivity analysis data
        if 'sensitivity_analysis' in self.simulation_data:
            for state_name, state_sensitivity in self.simulation_data['sensitivity_analysis'].items():
                for param_name, param_data in state_sensitivity.items():
                    sensitivity_df = pd.DataFrame(param_data['results'])
                    filename = f"{self.experiment_name}_{state_name}_{param_name}_sensitivity.csv"
                    sensitivity_df.to_csv(csv_dir / filename, index=False)
        
        # Save performance metrics
        if 'performance_metrics' in self.simulation_data:
            metrics_data = []
            for metric_name, metric_info in self.simulation_data['performance_metrics'].items():
                metrics_data.append({
                    'metric_name': metric_name,
                    'value': metric_info['value'],
                    'unit': metric_info['unit'],
                    'timestamp': metric_info['timestamp']
                })
            
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                df_metrics.to_csv(csv_dir / f"{self.experiment_name}_metrics.csv", index=False)
        
        self.logger.info(f"CSV data saved to: {csv_dir}")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of the simulation."""
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.simulation_data['experiment_info'].get('start_time'),
            'end_time': self.simulation_data['experiment_info'].get('end_time'),
            'description': self.simulation_data['experiment_info'].get('description', ''),
            'num_states_tested': len(self.simulation_data.get('results', {})),
            'num_errors': len(self.simulation_data.get('errors', [])),
            'num_warnings': len(self.simulation_data.get('warnings', [])),
            'performance_metrics': self.simulation_data.get('performance_metrics', {}),
            'results_summary': {}
        }
        
        # Add results summary
        for state_name, state_results in self.simulation_data.get('results', {}).items():
            if 'duopoly_solution' in state_results:
                solution = state_results['duopoly_solution']
                summary['results_summary'][state_name] = {
                    'insurer1_premium': solution['insurer1']['premium'],
                    'insurer1_profit': solution['insurer1']['expected_profit'],
                    'insurer2_premium': solution['insurer2']['premium'],
                    'insurer2_profit': solution['insurer2']['expected_profit'],
                    'total_profit': solution.get('total_profit', 0)
                }
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of the simulation."""
        summary = self.get_summary_report()
        
        print("\n" + "="*60)
        print(f"SIMULATION SUMMARY: {summary['experiment_name']}")
        print("="*60)
        
        if summary['description']:
            print(f"Description: {summary['description']}")
        
        print(f"Start Time: {summary['start_time']}")
        print(f"End Time: {summary['end_time']}")
        print(f"States Tested: {summary['num_states_tested']}")
        print(f"Errors: {summary['num_errors']}")
        print(f"Warnings: {summary['num_warnings']}")
        
        if summary['performance_metrics']:
            print("\nPerformance Metrics:")
            for metric, info in summary['performance_metrics'].items():
                print(f"  {metric}: {info['value']} {info['unit']}")
        
        if summary['results_summary']:
            print("\nResults Summary:")
            for state_name, results in summary['results_summary'].items():
                print(f"\n  {state_name}:")
                print(f"    Insurer 1 - Premium: {results['insurer1_premium']:.2f}, Profit: {results['insurer1_profit']:.2f}")
                print(f"    Insurer 2 - Premium: {results['insurer2_premium']:.2f}, Profit: {results['insurer2_profit']:.2f}")
                print(f"    Total Profit: {results['total_profit']:.2f}")
        
        print("="*60)


# Utility functions for easy logging integration
def create_logger(experiment_name: Optional[str] = None, **kwargs) -> SimulationLogger:
    """Create a new simulation logger with default settings."""
    return SimulationLogger(experiment_name=experiment_name, **kwargs)


def log_simulation_run(logger: SimulationLogger, 
                      params: Dict[str, Any],
                      state_spaces: List[Dict],
                      results: Dict[str, Any],
                      include_sensitivity: bool = False):
    """
    Log a complete simulation run.
    
    Args:
        logger: SimulationLogger instance
        params: Simulation parameters
        state_spaces: List of state space configurations
        results: Simulation results
        include_sensitivity: Whether sensitivity analysis was included
    """
    logger.log_experiment_start(f"Simulation with {len(state_spaces)} state spaces")
    logger.log_simulation_settings({
        'num_state_spaces': len(state_spaces),
        'include_sensitivity': include_sensitivity,
        'state_space_names': [s.get('name', 'unknown') for s in state_spaces]
    })
    logger.log_parameters(params)
    
    # Log results for each state space
    for state_config in state_spaces:
        state_name = state_config.get('name', 'default')
        if state_name in results:
            logger.log_duopoly_solution(results[state_name], state_name)
    
    logger.log_experiment_end("Simulation completed successfully")
    logger.print_summary() 