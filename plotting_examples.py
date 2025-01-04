import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from visualization.plotting import ModelPlotter
from simulation import SimulationSVCJ
from main import run_optimized_estimation
import os

def ensure_output_directory(output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def save_figure(fig, filename, output_dir='outputs'):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {filepath}")

def run_and_plot_examples(output_dir='outputs'):
    """Run estimation and create plots for results"""
    output_dir = ensure_output_directory(output_dir)
    plotter = ModelPlotter()

    # Run estimation
    print("Running model estimation...")
    results = run_optimized_estimation()
    
    # True parameters (from main.py)
    true_params = {
        'kappa': 0.96,
        'theta': 0.03,
        'sigma': 0.36,
        'rho': -0.92,
        'eta_s': 2.71,
        'eta_v': 0.62,
        'lmda': 0.72,
        'mu_s': 0.03,
        'sigma_s': 0.09,
        'eta_js': 0.06,
        'mu_v': 0.07,
        'eta_jv': 0.06,
        'rho_j': -0.74,
        'sigma_c': 3.10
    }

    # 1. Plot parameter estimation results
    print("\nPlotting parameter estimation results...")
    fig1 = plotter.plot_estimation_results(results, true_params)
    save_figure(fig1, 'parameter_estimation.png', output_dir)

    # 2. Plot filtered states
    print("Plotting filtered states...")
    data = {}
    if 'states' in results:
        data['true_states'] = results['states'].get('true_states')
        data['filtered_states'] = results['states'].get('filtered_states')
        fig2 = plotter.plot_filtered_states(data, results)
        save_figure(fig2, 'filtered_states.png', output_dir)

    # 3. Plot convergence diagnostics
    print("Plotting convergence diagnostics...")
    fig3 = plotter.plot_convergence_diagnostics(results)
    save_figure(fig3, 'convergence_diagnostics.png', output_dir)

    # 4. Plot option fit analysis if options data exists
    if 'options' in results:
        print("Plotting option fit analysis...")
        fig4 = plotter.plot_option_fit(results, results)
        if fig4:
            save_figure(fig4, 'option_fit.png', output_dir)

    # Print diagnostics summary
    print("\nEstimation Diagnostics Summary:")
    print(f"Mean acceptance rate: {np.mean(results['acceptance_rates']):.2%}")
    print("\nGelman-Rubin statistics (R̂):")
    for param, r_hat in results['r_hat'].items():
        print(f"{param}: {r_hat:.3f}")

def main():
    try:
        run_and_plot_examples()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()