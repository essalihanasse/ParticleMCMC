import numpy as np
from pipeline import EstimationPipeline
from particle_filter import ParticleFilter
from PMMH import OptimizedPMMH
from simulation import SimulationSVCJ
import time
import warnings
import os
from visualization.plotting import EnhancedModelPlotter
warnings.filterwarnings("ignore")

def create_output_directories():
    """Create output directories if they don't exist"""
    dirs = ['outputs', 'outputs/data', 'outputs/models', 'outputs/plots']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def run_optimized_estimation():
    print("Starting optimized SVCJ model estimation...")
    start_time = time.time()

    # Initialize model with simplified parameters
    model = SimulationSVCJ(
        kappa=0.96, theta=0.03, sigma=0.36, rho=-0.92,
        eta_s=2.71, eta_v=0.62, lmda=0.72,
        mu_s=0.03, sigma_s=0.09, eta_js=0.06,
        mu_v=0.07, eta_jv=0.06, rho_j=-0.74,
        sigma_c=3.10
    )

    # Simulation parameters
    S0 = 100.0
    V0 = 0.04
    r = 0.03
    delta = 0.02
    T = 20/252  # 20 trading days
    N = 50  

    print("\nSimulating price paths...")
    S, V = model.simulate_paths(S0, V0, r, delta, T, N)

    # Generate synthetic option data
    print("Generating option data...")
    option_data = []
    maturities = np.array([30, 60, 90, 180, 360])/360
    moneyness = np.array([0.9, 0.95, 1.0, 1.05, 1.1])

    # Generate options for specific dates
    sample_dates = np.linspace(0, len(S)-1, 5, dtype=int)
    
    for t in range(len(S)):
        if t in sample_dates:
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
        else:
            option_data.append([])

    # Initialize pipeline components
    pipeline = EstimationPipeline(
        particle_filter=ParticleFilter(num_particles=1000),
        pmmh=OptimizedPMMH(
            num_iterations=1000,
            num_chains=30,
            num_vertical=20,
            num_horizontal=1,
            use_orthogonal=True,
            burnin=100
        )
    )

    # Initial parameters
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
    print("\nRunning PMMH estimation...")
    results = pipeline.run(
        data={
            'prices': S,
            'options': option_data,
            'true_states': V
        },
        initial_params=initial_params
    )

    # Add additional data for plotting
    results['market_prices'] = [opt['price'] for day_opts in option_data if day_opts for opt in day_opts]
    results['model_prices'] = [model.compute_option_price(S[t], V[t], opt['K'], opt['tau'], r) 
                              for t, day_opts in enumerate(option_data) if day_opts 
                              for opt in day_opts]
    results['moneyness'] = [opt['K']/S[t] for t, day_opts in enumerate(option_data) if day_opts 
                           for opt in day_opts]
    results['maturities'] = [opt['tau'] for day_opts in option_data if day_opts 
                            for opt in day_opts]

    # Calculate execution time
    elapsed_time = time.time() - start_time
    results['execution_time'] = elapsed_time

    # Print parameter estimation results
    print("\nParameter Estimation Results:")
    print(f"{'Parameter':<10} {'True':<10} {'Estimated':<10} {'Error %':<10}")
    print("-" * 40)
    
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
        'mu_v': 0.07,
        'rho_j': -0.74
    }

    for param, true_value in true_params.items():
        est_value = results['posterior_means'].get(param, 0)
        error_pct = abs(est_value - true_value) / true_value * 100
        print(f"{param:<10} {true_value:<10.4f} {est_value:<10.4f} {error_pct:<10.1f}")

    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

    return results

def run_enhanced_analysis(results_dict):
    """Run complete analysis with organized plots"""
    plotter = EnhancedModelPlotter(base_output_dir='outputs/plots')
    metrics = plotter.create_analysis_report(results_dict)
    return metrics

def save_results(results, filename='model_results.npz'):
    """Save results to file"""
    save_path = os.path.join('outputs/data', filename)
    try:
        np.savez_compressed(
            save_path,
            posterior_means=results['posterior_means'],
            chains=results['chains'],
            acceptance_rates=results['acceptance_rates'],
            execution_time=results['execution_time']
        )
        print(f"\nResults saved to {save_path}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main():
    try:
        # Create output directories
        create_output_directories()
        
        # Run estimation
        results = run_optimized_estimation()
        
        # Save results
        save_results(results)
        
        # Run enhanced analysis with organized plotting
        metrics = run_enhanced_analysis(results)
        
        print("\nAnalysis completed successfully.")
        if metrics:
            print("\nSummary Metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()