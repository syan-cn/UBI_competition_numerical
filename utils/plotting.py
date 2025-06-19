"""
Plotting Module

Contains functions for visualizing simulation results and model outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_results(solver, solution: Dict, save_path: str = None):
    """Comprehensive plotting of simulation results with multiple visualization types."""
    if solution is None:
        print("No solution to plot.")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Premium comparison (Bar plot - Row 1, Column 1)
    phi1_1 = solution['insurer1']['phi1']
    phi1_2 = solution['insurer2']['phi1']
    
    bars = axes[0, 0].bar(['Insurer 1', 'Insurer 2'], [phi1_1, phi1_2], 
                         color=['blue', 'red'], alpha=0.7)
    axes[0, 0].set_ylabel(r'$\phi_1^i$ (Premium)')
    axes[0, 0].set_title('Optimal Premiums')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 2: Indemnity schedules (Bar plot - Row 1, Column 2)
    z_values = solver.z_values
    phi2_1 = solution['insurer1']['phi2_values']
    phi2_2 = solution['insurer2']['phi2_values']
    
    x_pos = np.arange(len(z_values))
    width = 0.35
    
    axes[0, 1].bar(x_pos - width/2, phi2_1, width, label='Insurer 1', alpha=0.7, color='blue')
    axes[0, 1].bar(x_pos + width/2, phi2_2, width, label='Insurer 2', alpha=0.7, color='red')
    axes[0, 1].set_xlabel('State Index')
    axes[0, 1].set_ylabel(r'$\phi_2^i(z)$ (Indemnity)')
    axes[0, 1].set_title('Discrete Indemnity Schedules')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([f'z={z:.1f}' for z in z_values])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hide the third subplot in row 1
    axes[0, 2].set_visible(False)
    
    # Plot 3: Action schedules comparison (Line plot - Row 2, Column 1)
    theta_grid = solver.theta_grid
    a1 = solution['insurer1']['a_schedule']
    a2 = solution['insurer2']['a_schedule']
    
    axes[1, 0].plot(theta_grid, a1, 'b-', linewidth=2, label='Insurer 1', marker='o')
    axes[1, 0].plot(theta_grid, a2, 'r--', linewidth=2, label='Insurer 2', marker='s')
    axes[1, 0].set_xlabel(r'$\theta$ (Risk Type)')
    axes[1, 0].set_ylabel(r'$a^i(\theta)$ (Action Level)')
    axes[1, 0].set_title('Optimal Action Schedules')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Expected utilities for different risk types (Line plot - Row 2, Column 2)
    V0_list = []
    V1_list = []
    V2_list = []
    for theta in theta_grid:
        # Compute expected utilities for each risk type using complete utility calculation
        a1_theta = np.interp(theta, theta_grid, a1)
        a2_theta = np.interp(theta, theta_grid, a2)
        
        V0 = solver.compute_reservation_utility(theta)
        V1 = solver.compute_expected_utility(a1_theta, phi1_1, phi2_1, solver.delta1, theta)
        V2 = solver.compute_expected_utility(a2_theta, phi1_2, phi2_2, solver.delta2, theta)
        V0_list.append(V0)
        V1_list.append(V1)
        V2_list.append(V2)
    
    axes[1, 1].plot(theta_grid, V0_list, 'k-', linewidth=2, label='No Insurance', marker='^')
    axes[1, 1].plot(theta_grid, V1_list, 'b-', linewidth=2, label='Insurer 1', marker='o')
    axes[1, 1].plot(theta_grid, V2_list, 'r--', linewidth=2, label='Insurer 2', marker='s')
    axes[1, 1].set_xlabel(r'$\theta$ (Risk Type)')
    axes[1, 1].set_ylabel('Utility')
    axes[1, 1].set_title('Expected Utility')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Choice probabilities (Line plot - Row 2, Column 3)
    P0_list = []
    P1_list = []
    P2_list = []
    
    for theta in theta_grid:
        # Compute choice probabilities for each risk type
        a1_theta = np.interp(theta, theta_grid, a1)
        a2_theta = np.interp(theta, theta_grid, a2)
        
        P0, P1, P2 = solver.compute_choice_probabilities(theta, a1_theta, phi1_1, phi2_1, a2_theta, phi1_2, phi2_2)
        P0_list.append(P0)
        P1_list.append(P1)
        P2_list.append(P2)
    
    axes[1, 2].plot(theta_grid, P0_list, 'k-', linewidth=2, label='No Insurance', marker='^')
    axes[1, 2].plot(theta_grid, P1_list, 'b-', linewidth=2, label='Insurer 1', marker='o')
    axes[1, 2].plot(theta_grid, P2_list, 'r--', linewidth=2, label='Insurer 2', marker='s')
    axes[1, 2].set_xlabel(r'$\theta$ (Risk Type)')
    axes[1, 2].set_ylabel('Choice Probability')
    axes[1, 2].set_title('Choice Probabilities')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, 1)  # Probabilities should be between 0 and 1

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()