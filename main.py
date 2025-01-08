import numpy as np
from pipeline import EstimationPipeline
from particle_filter import ParticleFilter
from PMMH import OptimizedPMMH
from simulation import SimulationSVCJ
from visualization.plotting import ModelPlotter
from diagnostics.model_diagnostics import ModelDiagnostics
import time
import warnings
import os
from tabulate import tabulate
warnings.filterwarnings("ignore")

def create_output_directories():
    """Create output directories if they don't exist"""
    dirs = ['outputs', 'outputs/data', 'outputs/models', 'outputs/plots']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def display_parameter_table(true_params, estimated_params, std_errors=None):
    """Display parameter estimation results in a nicely formatted table"""
    headers = ["Parameter", "True Value", "Estimate", "Std Error", "Error %"]
    rows = []
    for param, true_val in true_params.items():
        if param in estimated_params:
            est_val = estimated_params[param]
            std_err = std_errors.get(param, "N/A") if std_errors else "N/A"
            error_pct = abs(est_val - true_val) / true_val * 100 if true_val != 0 else 0
            rows.append([param, f"{true_val:.4f}", f"{est_val:.4f}", f"{std_err}", f"{error_pct:.1f}%"])
    
    print("\nParameter Estimation Results:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def display_diagnostics(diagnostics):
    """Display model diagnostics in a structured format"""
    print("\nModel Diagnostics:")
    print("-" * 50)
    
    # Convergence diagnostics
    if 'r_hat' in diagnostics:
        print("\nConvergence Diagnostics:")
        r_hat_values = diagnostics['r_hat']
        param_names = ['kappa', 'theta', 'sigma', 'rho', 'eta_s', 'eta_v', 'lmda', 
                      'mu_s', 'sigma_s', 'eta_js', 'mu_v', 'eta_jv', 'rho_j', 
                      'sigma_c', 'r', 'delta', 'V0']
                      
        # Create table format for R-hat values
        headers = ["Parameter", "R-hat"]
        rows = []
        
        # Handle case where r_hat is numpy array
        if isinstance(r_hat_values, np.ndarray):
            for param, r_hat in zip(param_names, r_hat_values):
                rows.append([param, f"{r_hat:.3f}"])
        else:
            # Handle case where r_hat is dictionary
            for param, r_hat in r_hat_values.items():
                rows.append([param, f"{r_hat:.3f}"])
                
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Option pricing errors
    if 'rmse' in diagnostics:
        print("\nOption Pricing Errors:")
        print(f"Overall RMSE: {diagnostics['rmse']:.4f}")
        if 'rmse_by_moneyness' in diagnostics:
            print("\nRMSE by Moneyness:")
            headers = ["Moneyness", "RMSE"]
            rows = []
            moneyness_bins = ['0.8-0.9', '0.9-1.0', '1.0-1.1', '1.1-1.2']
            for bin_name, rmse in zip(moneyness_bins, diagnostics['rmse_by_moneyness']):
                rows.append([bin_name, f"{rmse:.4f}"])
            print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Acceptance rates
    if 'acceptance_rates' in diagnostics:
        print("\nMCMC Acceptance Rates:")
        print(f"Mean acceptance rate: {diagnostics['acceptance_rates']:.2%}")
def run_model_estimation():
    """Run complete model estimation with comprehensive output"""
    print("Starting model estimation and analysis...")
    start_time = time.time()

    # Initialize model with annual parameters
    model = SimulationSVCJ(
        kappa=0.96,  # Annualize mean reversion
        theta=0.03,      # Keep variance level
        sigma=0.36,  # Annualize volatility of variance
        rho=-0.92,
        eta_s=2.71,
        eta_v=0.62,  # Annualize variance premium
        lmda=0.72,   # Annualize jump intensity
        mu_s=0.03,
        sigma_s=0.09,
        eta_js=0.06,
        mu_v=0.07,
        eta_jv=0.06, # Annualize jump variance premium
        rho_j=-0.74,
        sigma_c=3.10
    )

    # Simulation parameters
    S0, V0 = 100.0, 0.04
    r, delta = 0.03, 0.02
    T, N = 20/252, 252  # Use daily frequency

    print("\nSimulating price paths...")
    S, V = model.simulate_paths(S0, V0, r, delta, T, N)

    # Generate synthetic options
    print("Generating option data...")
    option_data = []
    maturities = np.array([30, 60, 90, 180, 360])/360
    moneyness = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
    
    for t in range(len(S)):
        daily_options = []
        for tau in maturities:
            for m in moneyness:
                K = m * S[t]
                daily_options.append({
                    'K': K,
                    'tau': tau,
                    'r': r,
                    'price': model.compute_option_price(S[t], V[t], K, tau, r)
                })
        option_data.append(daily_options)

    # Initialize components with proper scaling
    pipeline = EstimationPipeline(
        particle_filter=ParticleFilter(num_particles=100),
        pmmh=OptimizedPMMH(
            num_iterations=100,
            num_chains=5,
            num_vertical=10,
            num_horizontal=1,
            use_orthogonal=True,
            burnin=10
        )
    )

    # Initial parameters (properly scaled)
    initial_params = {
        'kappa': 1.0,
        'theta': 0.03,
        'sigma': 0.4,
        'rho': -0.7,
        'eta_s': 2.0,
        'eta_v': 0.5,
        'lmda': 0.5,
        'mu_s': 0.0,
        'sigma_s': 0.1,
        'eta_js': 0.05,
        'mu_v': 0.05,
        'eta_jv': 0.05,
        'rho_j': -0.5,
        'sigma_c': 3.0,
        'r': r,
        'delta': delta,
        'V0': V0
    }

    # Run estimation
    print("\nRunning MCMC estimation...")
    results = pipeline.run(
        data={'prices': S, 'options': option_data, 'true_states': V},
        initial_params=initial_params
    )

    # Add necessary information for plotting
    results.update({
        'param_names': list(initial_params.keys()),
        'true_params': {k: v for k, v in model.__dict__.items() if k in initial_params},
        'moneyness': moneyness,
        'maturity': maturities
    })

    # Initialize diagnostics and plotting
    diagnostics = ModelDiagnostics()
    plotter = ModelPlotter()

    # Compute diagnostics
    diag_results = {
        'r_hat': diagnostics.analyze_convergence(results['chains']),
        'acceptance_rates': np.mean(results.get('acceptance_rates', [])),
    }

    # Create plots
    print("\nGenerating analysis plots...")
    plotter.create_analysis_report(results)
   # Inside run_model_estimation(), after the estimation is complete:
    plotter.create_analysis_report(results)

# Prepare data for the heatmap
# We'll use the last time point as a snapshot
    t = -1  # Last time point
    S_t = S[t]  # Stock price at time t

# Get the true prices from our simulated data
    true_prices = [opt['price'] for opt in option_data[t]]

# Compute model prices using our estimated parameters
# We'll use the posterior means for our parameters
    estimated_params = results['posterior_means']

    # Create lists to store the moneyness, maturity values
    moneyness_values = []
    maturity_values = []
    model_prices = []

    # Compute model prices for each option
    for opt in option_data[t]:
        K = opt['K']
        tau = opt['tau']
        
        # Store moneyness and maturity
        moneyness_values.append(K/S_t)
        maturity_values.append(tau)
        
        # Compute model price using estimated parameters
        model_price = model.compute_option_price(
            S_t,
            estimated_params['V0'],  # Using estimated V0
            K,
            tau,
            opt['r']
        )
        model_prices.append(model_price)

    # Create the heatmap
    print("Creating option error heatmap...")
    plotter.plot_option_error_heatmap(
        true_prices,
        model_prices,
        moneyness_values,
        maturity_values
    )
    if 'true_prices' in results and 'model_prices' in results and 'moneyness' in results:
        print("Creating option error plots...")
        option_fit = plotter.plot_option_errors(
            true_prices=results['true_prices'],  # Using the true prices from simulation
            model_prices=results['model_prices'],  # Using the model-implied prices
            moneyness=results['moneyness'],
            maturity=results['maturity']
        )
        if option_fit is not None:
            plotter.save_figure(option_fit, 'option_errors.png')
    else:
        print("Skipping option error plots - required data not available in results")
        print("Available keys in results:", list(results.keys()))
    

    # Display results
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    display_parameter_table(model.__dict__, results['posterior_means'], 
                          results.get('posterior_std'))
    display_diagnostics(diag_results)

    return results, diag_results
def main():
    try:
        create_output_directories()
        results, diagnostics = run_model_estimation()
        print("\nAvailable data in results:")
        for key in results.keys():
            if isinstance(results[key], dict):
                print(f"{key}: {list(results[key].keys())}")
            elif isinstance(results[key], np.ndarray):
                print(f"{key}: Array of shape {results[key].shape}")
            else:
                print(f"{key}: {type(results[key])}")
        
        print("\nAnalysis completed successfully.")
        
    except Exception as e:
        print(f"\nAn error occurred during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()