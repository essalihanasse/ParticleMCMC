import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta

class ModelPlotter:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_estimation_results(self, results, true_params=None):
        param_names = list(results['posterior_means'].keys())
        n_params = len(param_names)
        
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 4*n_params))
        if n_params == 1:
            axes = [axes]
            
        for i, param in enumerate(param_names):
            chains = results['chains'][:, :, i]
            param_chains = chains.reshape(-1)
            
            # Plot histogram of posterior
            sns.histplot(param_chains, kde=True, ax=axes[i])
            
            # Add true value if available
            if true_params and param in true_params:
                axes[i].axvline(true_params[param], color='r', linestyle='--', 
                              label='True Value')
                
            # Add posterior mean
            axes[i].axvline(results['posterior_means'][param], color='g', 
                          linestyle='--', label='Posterior Mean')
            
            axes[i].set_title(f'{param} Posterior Distribution')
            axes[i].legend()
            
        plt.tight_layout()
        return fig

    def plot_filtered_states(self, data, results):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if data.get('true_states') is not None:
            times = np.arange(len(data['true_states']))
            ax.plot(times, data['true_states'], label='True Variance', 
                   color='black', alpha=0.6)
            
        if data.get('filtered_states') is not None:
            times = np.arange(len(data['filtered_states']))
            ax.plot(times, data['filtered_states'], label='Filtered Variance',
                   color='blue', alpha=0.8)
            
        ax.set_title('Variance State Estimation')
        ax.set_xlabel('Time')
        ax.set_ylabel('Variance')
        
        # Only add legend if we plotted something
        if data.get('true_states') is not None or data.get('filtered_states') is not None:
            ax.legend()
        
        return fig

    def plot_convergence_diagnostics(self, results):
        chains = results['chains']
        param_names = list(results['posterior_means'].keys())
        n_params = len(param_names)
        
        fig, axes = plt.subplots(n_params, 2, figsize=(15, 5*n_params))
        if n_params == 1:
            axes = axes.reshape(1, -1)
            
        for i, param in enumerate(param_names):
            # Trace plots
            for chain in range(chains.shape[0]):
                axes[i,0].plot(chains[chain,:,i], alpha=0.5)
            axes[i,0].set_title(f'{param} Trace Plot')
            axes[i,0].set_xlabel('Iteration')
            
            # Running mean plots
            running_means = np.cumsum(chains[:,:,i], axis=1) / \
                          np.arange(1, chains.shape[1] + 1)
            for chain in range(chains.shape[0]):
                axes[i,1].plot(running_means[chain], alpha=0.5)
            axes[i,1].set_title(f'{param} Running Mean')
            axes[i,1].set_xlabel('Iteration')
            
        plt.tight_layout()
        return fig

    def plot_option_fit(self, data, results):
        if 'options' not in data or not any(data['options']):
            return None
            
        # Extract option data
        options_data = []
        for date_options in data['options']:
            if not date_options:
                continue
            for opt in date_options:
                options_data.append({
                    'K': opt['K'],
                    'tau': opt['tau'],
                    'price': opt['price'],
                    'model_price': results.get('model_prices', {}).get(
                        (opt['K'], opt['tau']), None
                    )
                })
                
        if not options_data:
            return None
            
        df_options = pd.DataFrame(options_data)
        df_options['error'] = (df_options['model_price'] - df_options['price']) / \
                             df_options['price']
        
        # Create error heatmap
        pivot_errors = df_options.pivot_table(
            values='error', 
            index='tau',
            columns='K',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_errors, center=0, cmap='RdYlBu', ax=ax)
        ax.set_title('Relative Option Pricing Errors')
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Time to Maturity')
        
        return fig