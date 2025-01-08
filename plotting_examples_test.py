from visualization.plotting import EnhancedModelPlotter
import matplotlib.pyplot as plt
from main import run_optimized_estimation
import os
import numpy as np

def run_enhanced_analysis(output_dir='outputs'):
    """Run the full analysis pipeline with enhanced plotting"""
    
    # Ensure output directory exists
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Starting model estimation and analysis...")
    
    # Run model estimation
    results = run_optimized_estimation()
    
    # Initialize plotter
    plotter = EnhancedModelPlotter(output_dir)
    
    # 1. Plot parameter estimation results
    print("\nCreating parameter estimation plots...")
    plotter.plot_parameter_posteriors(
        results['chains'],
        results.get('true_params'),
        param_names=list(results['posterior_means'].keys())
    )
    
    # 2. Plot MCMC diagnostics
    print("Creating MCMC diagnostic plots...")
    plotter.plot_mcmc_diagnostics(
        results['chains'],
        param_names=list(results['posterior_means'].keys())
    )
    # In your run_enhanced_analysis function:

    # Plot parameter convergence in smaller chunks
    figs_conv = plotter.plot_parameter_convergence(
        results['chains'],
        param_names=list(results['posterior_means'].keys()),
        true_params=results.get('true_params'),
        n_rows=2, n_cols=2  # Adjust these numbers for desired layout
    )
    for i, fig in enumerate(figs_conv):
        plotter.save_figure(fig, f'parameter_convergence_{i+1}.png')

    # Plot hidden states if available
    if 'hidden_states' in results:
        fig_hidden = plotter.plot_hidden_states(
            true_states=results['hidden_states'].get('true'),
            estimated_states=results['hidden_states'].get('estimated'),
            dates=results.get('dates')
        )
        plotter.save_figure(fig_hidden, 'hidden_states.png')

    # More compact filtered states plot
    if 'states' in results:
        fig_filtered = plotter.plot_filtered_states_compact(
            true_states=results['states'].get('true_states'),
            filtered_states=results['states'].get('filtered_states'),
            confidence_bands=results['states'].get('confidence_bands')
        )
        plotter.save_figure(fig_filtered, 'filtered_states_compact.png')
    
    # 3. Plot filtered states and enhanced state space analysis
    if 'states' in results:
        print("Creating filtered states and state space analysis plots...")
        
        # Get the necessary state data
        true_states = results['states'].get('true_states')
        filtered_states = results['states'].get('filtered_states')
        observations = results.get('data', {}).get('prices', [])
        predicted_states = results['states'].get('predicted_states')
        predicted_std = results['states'].get('predicted_std')
        return_filtered = results['states'].get('return_filtered')
        joint_filtered = results['states'].get('joint_filtered')
        filtering_residuals = results['states'].get('residuals')

        # Original filtered states plot
        plotter.plot_filtered_states(true_states, filtered_states)

        # Enhanced state space plots
        # Plot filtered vs estimates (Image 1 style)
        fig1 = plotter.plot_filtered_vs_estimates(
            true_states=true_states,
            noisy_measurements=observations,
            filtered_estimate=filtered_states,
            prediction={'mean': predicted_states, 'std': predicted_std}
        )
        plotter.save_figure(fig1, 'filtered_vs_estimates.png')

        # Plot estimation comparison (Image 2 style)
        if return_filtered is not None and joint_filtered is not None:
            fig2 = plotter.plot_estimation_comparison(
                return_based={'filtered_estimate': return_filtered, 'true_state': true_states},
                joint_estimation={'filtered_estimate': joint_filtered}
            )
            plotter.save_figure(fig2, 'estimation_comparison.png')

        # Plot comprehensive diagnostics
        fig3 = plotter.plot_state_space_diagnostics(
            true_states=true_states,
            filtered_states=filtered_states,
            observations=observations,
            residuals=filtering_residuals
        )
        plotter.save_figure(fig3, 'state_space_diagnostics.png')
    
    # 4. Plot option fit analysis
    if 'options' in results:
        print("Creating option fit analysis plots...")
        plotter.plot_option_fit(
            results['options']['market_prices'],
            results['options']['model_prices'],
            results['options']['moneyness'],
            results['options']['maturities']
        )
    
    # 5. Plot model comparison if multiple models are estimated
    if 'model_comparison' in results:
        print("Creating model comparison plots...")
        plotter.plot_model_comparison(results['model_comparison'])
    
    print("\nAnalysis Summary:")
    print(f"Mean acceptance rate: {np.mean(results['acceptance_rates']):.2%}")
    print("\nGelman-Rubin statistics (R̂):")
    for param, r_hat in results['r_hat'].items():
        print(f"{param}: {r_hat:.3f}")
    
    print(f"\nAll plots have been saved to: {output_dir}/")
    return results
def create_state_space_analysis(results, output_dir='outputs'):
    """
    Create comprehensive state space analysis plots
    """
    plotter = EnhancedModelPlotter()
    
    # Get states from return-based and joint estimation
    if 'states' in results:
        # State space analysis
        fig_state = plotter.plot_state_space_analysis(
            true_states=results['states'].get('filtered_states'),  # Using filtered as reference
            filtered_states=results['states'].get('filtered_states'),
            observations=results.get('data', {}).get('prices', []),
            dates=None  # Add dates if available in your data
        )
        plotter.save_figure(fig_state, 'state_space_analysis.png')

    # Parameter convergence for all chains
    if 'chains' in results:
        fig_conv = plotter.plot_parameter_convergence(
            results['chains'],
            list(results['posterior_means'].keys())
        )
        plotter.save_figure(fig_conv, 'parameter_convergence.png')

    # Model diagnostics
    diagnostics = plotter.plot_mcmc_diagnostics(
        results['chains'],
        param_names=list(results['posterior_means'].keys())
    )
    plotter.save_figure(diagnostics, 'mcmc_diagnostics.png')

    # If option data is available
    if 'options' in results:
        option_fit = plotter.plot_option_fit(
            results['options'].get('market_prices', []),
            results['options'].get('model_prices', []),
            results['options'].get('moneyness', []),
            results['options'].get('maturities', [])
        )
        if option_fit is not None:
            plotter.save_figure(option_fit, 'option_fit.png')

    # Add acceptance rates plot
    if 'acceptance_rates' in results:
        fig_acc = plt.figure(figsize=(10, 6))
        plt.plot(results['acceptance_rates'])
        plt.title('MCMC Acceptance Rates')
        plt.xlabel('Chain')
        plt.ylabel('Acceptance Rate')
        plotter.save_figure(fig_acc, 'acceptance_rates.png')

    print(f"\nAll plots have been saved to: {output_dir}/")

    return {
        'gelman_rubin': results.get('r_hat', {}),
        'acceptance_rates': np.mean(results.get('acceptance_rates', [])),
    }
def main():
    try:
        results = run_enhanced_analysis()
        create_state_space_analysis(results=results)
        print("Analysis completed successfully.")
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
