import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class ModelPlotter:
    def __init__(self, output_dir='outputs/plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # plt.style.use('seaborn')
        
    def plot_parameter_chains(self, chains, param_names, true_params=None):
        """
        Plot MCMC chains for each parameter including a running mean overlay.
        This helps visualize both the raw chain behavior and convergence trends.
        
        Parameters:
        -----------
        chains : np.ndarray
            MCMC chains with shape (n_chains, n_iterations, n_params)
        param_names : list
            Names of parameters corresponding to the last dimension of chains
        true_params : dict, optional
            Dictionary of true parameter values for reference lines
        """
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4*n_params))
        
        if n_params == 1:
            axes = [axes]
            
        for i, (param, ax) in enumerate(zip(param_names, axes)):
            # Plot raw chains
            for chain in range(chains.shape[0]):
                chain_values = chains[chain, :, i]
                ax.plot(chain_values, alpha=0.3, label=f'Chain {chain+1}')
                
                # Calculate and plot running mean
                running_mean = np.cumsum(chain_values) / np.arange(1, len(chain_values) + 1)
                ax.plot(running_mean, color='red', linewidth=2, alpha=0.8, 
                    label='Running Mean' if chain == 0 else None)
                    
            if true_params and param in true_params:
                ax.axhline(y=true_params[param], color='green', linestyle='--', 
                        label='True Value')
                
            ax.set_title(f'Parameter: {param}')
            ax.set_xlabel('Iteration')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_chains.png'))
        plt.close()

    def plot_option_error_heatmap(self, true_prices, model_prices, moneyness, maturity):
        """
        Create a heatmap visualization of option pricing errors across different 
        moneyness-maturity combinations. This helps identify where the model performs 
        well or poorly.
        
        The heatmap shows relative pricing errors (model - true)/true, averaged across
        each moneyness-maturity combination. This relative measure helps us understand
        the model's performance regardless of the absolute price level of options.
        
        Parameters:
        -----------
        true_prices : np.ndarray or list
            True option prices from our simulation
        model_prices : np.ndarray or list
            Model-implied option prices computed using estimated parameters
        moneyness : np.ndarray or list
            Moneyness values (K/S) for each option
        maturity : np.ndarray or list
            Maturity values (in years) for each option
        """
        # Convert inputs to numpy arrays if they aren't already
        true_prices = np.asarray(true_prices)
        model_prices = np.asarray(model_prices)
        moneyness = np.asarray(moneyness)
        maturity = np.asarray(maturity)
        
        # Calculate relative errors
        relative_errors = np.abs((model_prices - true_prices) / true_prices) * 100  # Convert to percentage
        
        # Create a grid for the heatmap
        unique_moneyness = np.unique(moneyness)
        unique_maturity = np.unique(maturity)
        error_grid = np.zeros((len(unique_moneyness), len(unique_maturity)))
        
        # Fill the grid with average relative errors
        for i, m in enumerate(unique_moneyness):
            for j, t in enumerate(unique_maturity):
                mask = (moneyness == m) & (maturity == t)
                if np.any(mask):
                    error_grid[i, j] = np.mean(relative_errors[mask])
        
        # Create the heatmap with improved visualization
        plt.figure(figsize=(12, 8))
        sns.heatmap(error_grid, 
                    xticklabels=[f'{t:.2f}y' for t in unique_maturity],
                    yticklabels=[f'{m:.2f}' for m in unique_moneyness],
                    cmap='YlOrRd',
                    annot=True,
                    fmt='.1f',
                    cbar_kws={'label': 'Average Relative Error (%)'})
        
        plt.title('Option Pricing Errors by Moneyness and Maturity')
        plt.xlabel('Maturity (years)')
        plt.ylabel('Moneyness (K/S)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'option_error_heatmap.png'))
        plt.close()
    def plot_posterior_distributions(self, chains, param_names, true_params=None):
        """Plot posterior distributions for each parameter"""
        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4*n_params))
        
        if n_params == 1:
            axes = [axes]
            
        for i, (param, ax) in enumerate(zip(param_names, axes)):
            for chain in range(chains.shape[0]):
                sns.kdeplot(chains[chain, :, i], ax=ax, alpha=0.5)
                
            if true_params and param in true_params:
                ax.axvline(x=true_params[param], color='r', linestyle='--', label='True')
                
            ax.set_title(f'Posterior Distribution: {param}')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'posterior_distributions.png'))
        plt.close()
        
  
    def create_analysis_report(self, results):
        """Create complete analysis report with all plots"""
        if 'chains' in results and 'param_names' in results:
            self.plot_parameter_chains(
                results['chains'],
                results['param_names'],
                results.get('true_params')
            )
            
            self.plot_posterior_distributions(
                results['chains'],
                results['param_names'],
                results.get('true_params')
            )
            
        if all(k in results for k in ['true_prices', 'model_prices', 'moneyness', 'maturity']):
            self.plot_option_errors(
                results['true_prices'],
                results['model_prices'],
                results['moneyness'],
                results['maturity']
            )