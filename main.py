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
    dirs = ['outputs', 'outputs/plots', 'outputs/diagnostics']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def display_parameter_table(true_params, estimated_params, std_errors=None):
    """Display parameter estimation results with enhanced formatting"""

    headers = ["Parameter", "True Value", "Estimate", "Std Error", "Error %", "95% CI"]
    rows = []
    for param, true_val in true_params.items():
        if param in estimated_params:
            est_val = estimated_params[param]
            std_err = std_errors.get(param, "N/A") if std_errors else "N/A"
            error_pct = abs(est_val - true_val) / true_val * 100 if true_val != 0 else 0

            # Calculate 95% confidence interval
            if std_err != "N/A":
                ci_lower = est_val - 1.96 * std_err
                ci_upper = est_val + 1.96 * std_err
                ci = f"({ci_lower:.4f}, {ci_upper:.4f})"
            else:
                ci = "N/A"

            rows.append([
                param, 
                f"{true_val:.4f}", 
                f"{est_val:.4f}", 
                f"{std_err}", 
                f"{error_pct:.1f}%",
                ci
            ])

    print("\nParameter Estimation Results:")
    print(tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".4f"))


def display_diagnostics(diagnostics):
    """Display enhanced model diagnostics"""
    print("\nModel Diagnostics:")
    print("-" * 50)
    
    # Convergence diagnostics
    if 'r_hat' in diagnostics:
        print("\nGelman-Rubin Convergence Diagnostics (R̂):")
        r_hat_values = diagnostics['r_hat']
        param_names = ['kappa', 'theta', 'sigma', 'rho', 'eta_s', 'eta_v', 'lmda', 
                      'mu_s', 'sigma_s', 'eta_js', 'mu_v', 'eta_jv', 'rho_j', 
                      'sigma_c', 'r', 'delta', 'V0']
        
        headers = ["Parameter", "R-hat", "Status"]
        rows = []
        
        for param, r_hat in zip(param_names, r_hat_values):
            status = "✓ Converged" if r_hat < 1.1 else "⚠ Not converged"
            rows.append([param, f"{r_hat:.3f}", status])
                
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Option pricing errors
    if 'rmse' in diagnostics:
        print("\nOption Pricing Errors:")
        print(f"Overall RMSE: {diagnostics['rmse']:.4f}")
        
        if 'rmse_by_moneyness' in diagnostics:
            print("\nRMSE by Moneyness:")
            headers = ["Moneyness", "RMSE", "Sample Size"]
            rows = []
            moneyness_bins = ['0.8-0.9', '0.9-1.0', '1.0-1.1', '1.1-1.2']
            for bin_name, rmse, size in zip(moneyness_bins, 
                                          diagnostics['rmse_by_moneyness'],
                                          diagnostics['sample_sizes']):
                rows.append([bin_name, f"{rmse:.4f}", size])
            print(tabulate(rows, headers=headers, tablefmt="grid"))

    # MCMC diagnostics
    if 'acceptance_rates' in diagnostics:
        print("\nMCMC Diagnostics:")
        print(f"Mean acceptance rate: {diagnostics['acceptance_rates']:.2%}")
        if 'effective_sample_size' in diagnostics:
            print(f"Effective sample size: {diagnostics['effective_sample_size']:.0f}")

def parse_arguments():
    """Parse command line arguments with improved defaults"""
    parser = argparse.ArgumentParser(description='Run SVCJ model estimation with custom parameters')
    
    # Pipeline configuration
    parser.add_argument('--num-particles', type=int, default=1000,
                       help='Number of particles for the particle filter')
    parser.add_argument('--mcmc-iterations', type=int, default=500,
                       help='Number of MCMC iterations')
    parser.add_argument('--num-chains', type=int, default=5,
                       help='Number of parallel MCMC chains')
    parser.add_argument('--burnin', type=int, default=100,
                       help='Number of burn-in iterations')
    parser.add_argument('--vertical-moves', type=int, default=10,
                       help='Number of vertical moves in PMMH')
    
    # Time horizon configuration
    parser.add_argument('--time-horizon', type=float, default=20/252,
                       help='Time horizon T in years (default: 20 trading days)')
    parser.add_argument('--time-steps', type=int, default=252,
                       help='Number of time steps N')
    
    # Additional parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with additional logging')
    
    return parser.parse_args()

def main():
    """Enhanced main function with improved error handling and diagnostics"""
    try:
        # Set random seed for reproducibility
        args = parse_arguments()
        np.random.seed(args.seed)
        
        # Create output directories
        create_output_directories()
        
        # Initialize pipeline components
        pipeline = EstimationPipeline(
            particle_filter=ParticleFilter(
                num_particles=args.num_particles,
                num_quantiles=24  # Increased from 12
            ),
            pmmh=OptimizedPMMH(
                num_iterations=args.mcmc_iterations,
                num_chains=args.num_chains,
                num_vertical=args.vertical_moves,
                num_horizontal=1,
                use_orthogonal=True,
                burnin=args.burnin
            )
        )
        
        # Initialize model with annual parameters
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

        # Simulation parameters
        S0, V0 = 100.0, 0.04
        r, delta = 0.03, 0.02
        T, N = args.time_horizon, args.time_steps

        print(f"\nRunning simulation with:")
        print(f"Time horizon (T): {T:.4f} years")
        print(f"Number of time steps (N): {N}")
        print(f"Number of particles: {args.num_particles}")
        print(f"MCMC iterations: {args.mcmc_iterations}")
        print(f"Number of chains: {args.num_chains}")
        
        # Simulate price paths
        print("\nSimulating price paths...")
        S, V = model.simulate_paths(S0, V0, r, delta, T, N)

        # Generate synthetic options with improved coverage
        print("Generating option data...")
        option_data = []
        maturities = np.array([30, 60, 90, 180, 360])/360
        moneyness = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15])
        
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

        # Initial parameters with reasonable bounds
        initial_params = {
            'kappa': 1.0, 'theta': 0.03, 'sigma': 0.4, 'rho': -0.7,
            'eta_s': 2.0, 'eta_v': 0.5, 'lmda': 0.5, 'mu_s': 0.0,
            'sigma_s': 0.1, 'eta_js': 0.05, 'mu_v': 0.05, 'eta_jv': 0.05,
            'rho_j': -0.5, 'sigma_c': 3.0, 'r': r, 'delta': delta, 'V0': V0
        }

        # Run estimation with progress tracking
        print("\nRunning MCMC estimation...")
        start_time = time.time()
        results = pipeline.run(
            data={'prices': S, 'options': option_data, 'true_states': V},
            initial_params=initial_params
        )
        estimation_time = time.time() - start_time
        print(f"\nEstimation completed in {estimation_time:.2f} seconds")

        # Add necessary information for analysis
        results.update({
            'param_names': list(initial_params.keys()),
            'true_params': {k: v for k, v in model.__dict__.items() if k in initial_params},
            'moneyness': moneyness,
            'maturity': maturities,
            'estimation_time': estimation_time
        })

        # Initialize diagnostics and plotting
        diagnostics = ModelDiagnostics()
        plotter = ModelPlotter()

        # Compute comprehensive diagnostics
        diag_results = diagnostics.compute_all_diagnostics(results)

        # Display results and diagnostics
        display_parameter_table(model.__dict__, results['posterior_means'], 
                              results.get('posterior_std'))
        display_diagnostics(diag_results)

        print("\nGenerating analysis plots...")
        plotter.create_analysis_report(results)

        print("\nAnalysis completed successfully.")
        return results, diag_results

    except Exception as e:
        print(f"\nAn error occurred during analysis: {str(e)}")
        if args.debug:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        raise

if __name__ == "__main__":
    main()