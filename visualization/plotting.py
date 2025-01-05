import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from statsmodels.tsa.stattools import acf
import pandas as pd

class EnhancedModelPlotter:
    def __init__(self, base_output_dir='outputs'):
        """
        Initialize plotter with organized folder structure
        """
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.folders = {
            'state_space': 'state_space_analysis',
            'parameters': 'parameter_analysis',
            'diagnostics': 'mcmc_diagnostics',
            'options': 'option_analysis',
            'metrics': 'performance_metrics'
        }
        self.setup_directories()
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def setup_directories(self):
        """
        Create organized folder structure for outputs
        """
        # Create main output directory with timestamp
        self.run_dir = os.path.join(self.base_output_dir, f'run_{self.timestamp}')
        
        # Create subdirectories
        for folder in self.folders.values():
            folder_path = os.path.join(self.run_dir, folder)
            os.makedirs(folder_path, exist_ok=True)

    def save_figure(self, fig, filename, subfolder=None):
        """
        Save figure to appropriate subfolder
        """
        if subfolder:
            save_path = os.path.join(self.run_dir, self.folders.get(subfolder, ''), filename)
        else:
            save_path = os.path.join(self.run_dir, filename)
            
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure to {save_path}")

    def plot_state_space_analysis(self, true_states, filtered_states, observations, dates=None):
        """
        Create comprehensive state space analysis plots
        """
        try:
            # State comparison plot
            fig_state = plt.figure(figsize=(12, 6))
            times = dates if dates is not None else np.arange(len(true_states))
            plt.plot(times, true_states, 'k-', label='True State', alpha=0.6)
            plt.plot(times, filtered_states, 'b--', label='Filtered State', alpha=0.8)
            plt.title('State Filtering Results')
            plt.legend()
            self.save_figure(fig_state, 'state_comparison.png', 'state_space')

            # Error analysis
            fig_error = plt.figure(figsize=(15, 5))
            errors = filtered_states - true_states
            plt.subplot(131)
            plt.plot(times, errors, 'r.')
            plt.title('Filtering Errors')
            
            plt.subplot(132)
            plt.hist(errors, bins=50, density=True)
            plt.title('Error Distribution')
            
            plt.subplot(133)
            error_acf = acf(errors, nlags=40)
            plt.bar(range(len(error_acf)), error_acf)
            plt.title('Error ACF')
            self.save_figure(fig_error, 'error_analysis.png', 'state_space')

            return True

        except Exception as e:
            print(f"Error in state space analysis: {str(e)}")
            return False

    def plot_parameter_convergence(self, chains, param_names, burn_in=1000):
        """
        Plot parameter convergence diagnostics
        """
        try:
            for i, param in enumerate(param_names):
                # Trace plots
                fig_trace = plt.figure(figsize=(12, 5))
                for chain in chains:
                    plt.plot(chain[burn_in:, i], alpha=0.5)
                plt.title(f'{param} Trace Plot')
                self.save_figure(fig_trace, f'{param}_trace.png', 'parameters')

                # Posterior density
                fig_density = plt.figure(figsize=(12, 5))
                for chain in chains:
                    sns.kdeplot(chain[burn_in:, i], alpha=0.5)
                plt.title(f'{param} Posterior Density')
                self.save_figure(fig_density, f'{param}_density.png', 'parameters')

            return True

        except Exception as e:
            print(f"Error in parameter convergence plots: {str(e)}")
            return False

    def plot_mcmc_diagnostics(self, chains, param_names):
        """
        Plot MCMC diagnostics
        """
        try:
            # Gelman-Rubin statistics
            r_hats = [self._compute_gelman_rubin(chains[:,:,i]) for i in range(len(param_names))]
            
            fig_gelman = plt.figure(figsize=(10, 6))
            plt.bar(param_names, r_hats)
            plt.xticks(rotation=45)
            plt.title('Gelman-Rubin Statistics')
            plt.axhline(y=1.1, color='r', linestyle='--', label='Threshold')
            plt.legend()
            self.save_figure(fig_gelman, 'gelman_rubin.png', 'diagnostics')

            # Chain mixing plots
            for i, param in enumerate(param_names):
                fig_mix = plt.figure(figsize=(12, 5))
                chain_means = np.mean(chains[:,:,i], axis=1)
                chain_stds = np.std(chains[:,:,i], axis=1)
                
                plt.errorbar(range(len(chain_means)), chain_means, 
                           yerr=chain_stds, fmt='o', alpha=0.5)
                plt.title(f'{param} Chain Mixing')
                self.save_figure(fig_mix, f'{param}_mixing.png', 'diagnostics')

            return True

        except Exception as e:
            print(f"Error in MCMC diagnostics: {str(e)}")
            return False

    def plot_option_fit(self, market_prices, model_prices, moneyness, maturities):
        """
        Plot option pricing fit analysis
        """
        try:
            # Price comparison
            fig_price = plt.figure(figsize=(10, 6))
            plt.scatter(market_prices, model_prices, alpha=0.5)
            max_price = max(max(market_prices), max(model_prices))
            plt.plot([0, max_price], [0, max_price], 'r--')
            plt.xlabel('Market Prices')
            plt.ylabel('Model Prices')
            plt.title('Model vs Market Prices')
            self.save_figure(fig_price, 'price_comparison.png', 'options')

            # Error analysis by moneyness and maturity
            fig_error = plt.figure(figsize=(12, 5))
            errors = model_prices - market_prices
            
            plt.subplot(121)
            plt.scatter(moneyness, errors, alpha=0.5)
            plt.xlabel('Moneyness')
            plt.ylabel('Pricing Error')
            plt.title('Errors by Moneyness')
            
            plt.subplot(122)
            plt.scatter(maturities, errors, alpha=0.5)
            plt.xlabel('Maturity')
            plt.ylabel('Pricing Error')
            plt.title('Errors by Maturity')
            
            self.save_figure(fig_error, 'error_analysis.png', 'options')

            return True

        except Exception as e:
            print(f"Error in option fit plots: {str(e)}")
            return False

    def create_analysis_report(self, results):
        """
        Create comprehensive analysis with all plots
        """
        try:
            # State space analysis
            if all(k in results for k in ['true_states', 'filtered_states']):
                self.plot_state_space_analysis(
                    results['true_states'],
                    results['filtered_states'],
                    results.get('observations', None)
                )

            # Parameter analysis
            if 'chains' in results and 'posterior_means' in results:
                self.plot_parameter_convergence(
                    results['chains'],
                    list(results['posterior_means'].keys())
                )

            # MCMC diagnostics
            if 'chains' in results:
                self.plot_mcmc_diagnostics(
                    results['chains'],
                    list(results['posterior_means'].keys())
                )

            # Option analysis
            if all(k in results for k in ['market_prices', 'model_prices', 'moneyness', 'maturities']):
                self.plot_option_fit(
                    results['market_prices'],
                    results['model_prices'],
                    results['moneyness'],
                    results['maturities']
                )

            # Save summary metrics
            metrics = {
                'gelman_rubin': results.get('r_hat', {}),
                'acceptance_rates': np.mean(results.get('acceptance_rates', [])),
                'rmse': results.get('rmse', None),
                'timestamp': self.timestamp
            }
            
            metrics_path = os.path.join(self.run_dir, self.folders['metrics'], 'summary_metrics.txt')
            with open(metrics_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")

            print(f"Analysis complete. Results saved in {self.run_dir}")
            return metrics

        except Exception as e:
            print(f"Error in analysis report creation: {str(e)}")
            return None

    def _compute_gelman_rubin(self, chains):
        """
        Compute Gelman-Rubin convergence diagnostic
        """
        try:
            M = chains.shape[0]  # Number of chains
            N = chains.shape[1]  # Length of each chain
            
            # Between-chain variance
            chain_means = np.mean(chains, axis=1)
            overall_mean = np.mean(chain_means)
            B = N * np.var(chain_means, ddof=1)
            
            # Within-chain variance
            W = np.mean(np.var(chains, axis=1, ddof=1))
            
            # Calculate R-hat
            var_est = ((N - 1) * W / N + B / N)
            R_hat = np.sqrt(var_est / W)
            
            return R_hat
            
        except Exception as e:
            print(f"Error computing Gelman-Rubin statistic: {str(e)}")
            return np.nan