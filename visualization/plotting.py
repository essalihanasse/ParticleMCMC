import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from statsmodels.tsa.stattools import acf

class EnhancedModelPlotter:
    def __init__(self, base_output_dir='outputs'):
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
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def setup_directories(self):
        self.run_dir = os.path.join(self.base_output_dir, f'run_{self.timestamp}')
        for folder in self.folders.values():
            os.makedirs(os.path.join(self.run_dir, folder), exist_ok=True)

    def save_figure(self, fig, filename, subfolder=None):
        save_path = os.path.join(self.run_dir, self.folders.get(subfolder, ''), filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved figure to {save_path}")

    def create_analysis_report(self, results):
        try:
            if 'chains' in results and 'posterior_means' in results:
                param_names = list(results['posterior_means'].keys())
                true_params = results.get('true_params')
                burn_in = results.get('burn_in', int(0.5 * results['chains'].shape[1]))
                
                fig_traces = self.plot_parameter_traces(
                    results['chains'],
                    param_names,
                    true_params,
                    burn_in
                )
                self.save_figure(fig_traces, 'parameter_traces.png', 'parameters')
                
                fig_dist = self.plot_parameter_distributions(
                    results['chains'],
                    param_names,
                    true_params,
                    burn_in
                )
                self.save_figure(fig_dist, 'parameter_distributions.png', 'parameters')
                
                fig_conv = self.plot_convergence_diagnostics(
                    results['chains'],
                    param_names,
                    burn_in
                )
                self.save_figure(fig_conv, 'convergence_diagnostics.png', 'diagnostics')

            if all(k in results for k in ['true_states', 'filtered_states']):
                fig_state = self.plot_state_space_analysis(
                    results['true_states'],
                    results['filtered_states'],
                    results.get('observations', None),
                    results.get('dates', None)
                )
                self.save_figure(fig_state, 'state_space_analysis.png', 'state_space')

            if all(k in results for k in ['market_prices', 'model_prices', 'moneyness', 'maturities']):
                fig_opt = self.plot_option_fit(
                    results['market_prices'],
                    results['model_prices'],
                    results['moneyness'],
                    results['maturities']
                )
                self.save_figure(fig_opt, 'option_fit.png', 'options')

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

            return metrics

        except Exception as e:
            print(f"Error in analysis report creation: {str(e)}")
            return None

    def plot_parameter_traces(self, chains, param_names, true_params=None, burn_in=None):
        n_params = len(param_names)
        n_cols = 2
        n_rows = (n_params + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()

        for i, param in enumerate(param_names):
            ax = axes[i]
            for chain in range(chains.shape[0]):
                chain_data = chains[chain, burn_in:, i] if burn_in else chains[chain, :, i]
                ax.plot(chain_data, alpha=0.5, label=f'Chain {chain+1}', color=self.colors[chain])
            
            if true_params and param in true_params:
                ax.axhline(y=true_params[param], color='r', linestyle='--', label='True Value')
            
            ax.set_title(f'{param} Trace')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.legend()

        for i in range(n_params, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        return fig

    def plot_parameter_distributions(self, chains, param_names, true_params=None, burn_in=None):
        n_params = len(param_names)
        n_cols = 2
        n_rows = (n_params + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()

        for i, param in enumerate(param_names):
            ax = axes[i]
            for chain in range(chains.shape[0]):
                chain_data = chains[chain, burn_in:, i] if burn_in else chains[chain, :, i]
                sns.kdeplot(chain_data, ax=ax, alpha=0.5, label=f'Chain {chain+1}', color=self.colors[chain])
            
            if true_params and param in true_params:
                ax.axvline(x=true_params[param], color='r', linestyle='--', label='True Value')
            
            ax.set_title(f'{param} Posterior Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()

        for i in range(n_params, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        return fig

    def plot_convergence_diagnostics(self, chains, param_names, burn_in=None):
        n_chains, n_iters, n_params = chains.shape
        window_size = n_iters // 10
        running_means = np.zeros((n_params, n_iters-window_size+1))
        r_hats = []
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Calculate and plot running means
        for i in range(n_params):
            chain_data = chains[:, burn_in:, i] if burn_in else chains[:, :, i]
            mean_chain = np.mean(chain_data, axis=0)
            
            for j in range(len(mean_chain) - window_size + 1):
                running_means[i,j] = np.mean(mean_chain[j:j+window_size])
            ax1.plot(running_means[i,:], label=param_names[i], color=self.colors[i])
            
            r_hat = self._compute_gelman_rubin(chain_data)
            r_hats.append(r_hat)
        
        ax1.set_title('Running Means')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Mean Value')
        ax1.legend()
        
        ax2.bar(param_names, r_hats, color=self.colors[:len(param_names)])
        ax2.axhline(y=1.1, color='r', linestyle='--', label='Convergence Threshold')
        ax2.set_title('Gelman-Rubin Statistics (R̂)')
        ax2.set_xlabel('Parameter')
        ax2.set_ylabel('R̂')
        plt.xticks(rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        return fig

    def plot_state_space_analysis(self, true_states, filtered_states, observations, dates=None):
        times = dates if dates is not None else np.arange(len(true_states))
        
        fig = plt.figure(figsize=(15, 12))
        
        # State trajectories
        ax1 = plt.subplot(311)
        ax1.plot(times, true_states, 'k-', label='True State', alpha=0.6)
        ax1.plot(times, filtered_states, 'b--', label='Filtered State', alpha=0.8)
        ax1.set_title('State Filtering Results')
        ax1.legend()
        
        # Filtering errors
        errors = filtered_states - true_states
        ax2 = plt.subplot(312)
        ax2.plot(times, errors, 'r.', alpha=0.5)
        ax2.set_title('Filtering Errors')
        ax2.set_ylabel('Error')
        
        # Error distribution and ACF
        ax3 = plt.subplot(313)
        sns.histplot(errors, stat='density', kde=True, ax=ax3)
        ax3.set_title('Error Distribution')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Density')
        
        plt.tight_layout()
        return fig

    def plot_option_fit(self, market_prices, model_prices, moneyness, maturities):
        fig = plt.figure(figsize=(15, 12))
        
        # Price comparison
        ax1 = plt.subplot(221)
        ax1.scatter(market_prices, model_prices, alpha=0.5, color=self.colors[0])
        max_price = max(max(market_prices), max(model_prices))
        ax1.plot([0, max_price], [0, max_price], 'r--')
        ax1.set_xlabel('Market Prices')
        ax1.set_ylabel('Model Prices')
        ax1.set_title('Model vs Market Prices')
        
        # Errors by moneyness
        errors = model_prices - market_prices
        ax2 = plt.subplot(222)
        ax2.scatter(moneyness, errors, alpha=0.5, color=self.colors[1])
        ax2.set_xlabel('Moneyness')
        ax2.set_ylabel('Pricing Error')
        ax2.set_title('Errors by Moneyness')
        
        # Errors by maturity
        ax3 = plt.subplot(223)
        ax3.scatter(maturities, errors, alpha=0.5, color=self.colors[2])
        ax3.set_xlabel('Maturity')
        ax3.set_ylabel('Pricing Error')
        ax3.set_title('Errors by Maturity')
        
        # Error distribution
        ax4 = plt.subplot(224)
        sns.histplot(errors, stat='density', kde=True, color=self.colors[3], ax=ax4)
        ax4.set_title('Error Distribution')
        ax4.set_xlabel('Pricing Error')
        ax4.set_ylabel('Density')
        
        plt.tight_layout()
        return fig

    def _compute_gelman_rubin(self, chains):
        """Compute Gelman-Rubin convergence diagnostic"""
        # Reshape chains if needed
        if len(chains.shape) == 1:
            chains = chains.reshape(-1, 1)
        
        n_chains = chains.shape[0]
        n_samples = chains.shape[1]
        
        # Calculate between-chain variance B
        chain_means = np.mean(chains, axis=1)
        overall_mean = np.mean(chain_means)
        B = n_samples * np.var(chain_means, ddof=1)
        
        # Calculate within-chain variance W
        W = np.mean(np.var(chains, axis=1, ddof=1))
        
        # Calculate R-hat
        var_est = ((n_samples - 1) * W / n_samples + B / n_samples)
        R_hat = np.sqrt(var_est / W)
        
        return R_hat