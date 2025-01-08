import argparse
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
def create_output_directories():
    """Create output directories if they don't exist"""
    dirs = ['outputs','outputs/plots']
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
def parse_arguments():
    """Parse command line arguments for model configuration."""
    parser = argparse.ArgumentParser(description='Run SVCJ model estimation with custom parameters')
    
    # Pipeline configuration arguments
    parser.add_argument('--num-particles', type=int, default=100,
                       help='Number of particles for the particle filter')
    parser.add_argument('--mcmc-iterations', type=int, default=100,
                       help='Number of MCMC iterations')
    parser.add_argument('--num-chains', type=int, default=5,
                       help='Number of parallel MCMC chains')
    parser.add_argument('--burnin', type=int, default=10,
                       help='Number of burn-in iterations')
    parser.add_argument('--vertical-moves', type=int, default=10,
                       help='Number of vertical moves in PMMH')
    
    # Time horizon configuration
    parser.add_argument('--time-horizon', type=float, default=20/252,
                       help='Time horizon T in years (default: 20 trading days)')
    parser.add_argument('--time-steps', type=int, default=252,
                       help='Number of time steps N')
    
    return parser.parse_args()

def main():
    """Main function with configurable pipeline parameters and time horizon."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create output directories
        create_output_directories()
        
        # Initialize pipeline components with command line arguments
        pipeline = EstimationPipeline(
            particle_filter=ParticleFilter(num_particles=args.num_particles),
            pmmh=OptimizedPMMH(
                num_iterations=args.mcmc_iterations,
                num_chains=args.num_chains,
                num_vertical=args.vertical_moves,
                num_horizontal=1,
                use_orthogonal=True,
                burnin=args.burnin
            )
        )
        
        # Initialize model with annual parameters (same as before)
        model = SimulationSVCJ(
            kappa=0.96,
            theta=0.03,
            sigma=0.36,
            rho=-0.92,
            eta_s=2.71,
            eta_v=0.62,
            lmda=0.72,
            mu_s=0.03,
            sigma_s=0.09,
            eta_js=0.06,
            mu_v=0.07,
            eta_jv=0.06,
            rho_j=-0.74,
            sigma_c=3.10
        )

        # Simulation parameters with configurable time horizon
        S0, V0 = 100.0, 0.04
        r, delta = 0.03, 0.02
        T, N = args.time_horizon, args.time_steps

        print(f"\nRunning simulation with:")
        print(f"Time horizon (T): {T:.4f} years")
        print(f"Number of time steps (N): {N}")
        print(f"Number of particles: {args.num_particles}")
        print(f"MCMC iterations: {args.mcmc_iterations}")
        print(f"Number of chains: {args.num_chains}")
        
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

        # Initial parameters (same as before)
        initial_params = {
            'kappa': 1.0, 'theta': 0.03, 'sigma': 0.4, 'rho': -0.7,
            'eta_s': 2.0, 'eta_v': 0.5, 'lmda': 0.5, 'mu_s': 0.0,
            'sigma_s': 0.1, 'eta_js': 0.05, 'mu_v': 0.05, 'eta_jv': 0.05,
            'rho_j': -0.5, 'sigma_c': 3.0, 'r': r, 'delta': delta, 'V0': V0
        }

        # Run estimation with configured pipeline
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

        # Display results
        display_parameter_table(model.__dict__, results['posterior_means'], 
                              results.get('posterior_std'))
        display_diagnostics(diag_results)

        print("\nGenerating analysis plots...")
        plotter.create_analysis_report(results)

        print("\nAnalysis completed successfully.")
        return results, diag_results

    except Exception as e:
        print(f"\nAn error occurred during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()