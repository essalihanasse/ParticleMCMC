import numpy as np
from scipy import stats

class ModelDiagnostics:
    """
    A comprehensive diagnostics class for MCMC estimation of stochastic volatility models.
    Provides methods for convergence analysis, parameter estimation quality,
    and option pricing accuracy.
    """
    
    def __init__(self):
        self.results = None
        
    def compute_all_diagnostics(self, results):
        """
        Compute comprehensive diagnostics for the estimation results.
        
        Args:
            results (dict): Dictionary containing estimation results including
                          chains, acceptance rates, and other metadata
        
        Returns:
            dict: Dictionary containing all diagnostic measures
        """
        self.results = results
        diagnostics = {}
        
        # Compute Gelman-Rubin convergence diagnostics
        diagnostics['r_hat'] = self._compute_gelman_rubin()
        
        # Compute effective sample sizes
        diagnostics['effective_sample_size'] = self._compute_effective_sample_size()
        
        # Compute acceptance rates
        diagnostics['acceptance_rates'] = self._compute_acceptance_rates()
        
        # Compute option pricing errors if options data is available
        if 'options' in results:
            pricing_diagnostics = self._compute_option_pricing_errors()
            diagnostics.update(pricing_diagnostics)
        
        # Compute parameter estimation accuracy
        if 'true_params' in results:
            param_diagnostics = self._compute_parameter_diagnostics()
            diagnostics.update(param_diagnostics)
        
        return diagnostics
    
    def _compute_gelman_rubin(self):
        """
        Compute Gelman-Rubin convergence diagnostic (R-hat) for all parameters.
        """
        chains = self.results['chains']
        n_chains, n_iterations, n_params = chains.shape
        
        # Use only second half of chains for convergence diagnostics
        start_idx = n_iterations // 2
        chains = chains[:, start_idx:, :]
        
        r_hat = np.zeros(n_params)
        
        for j in range(n_params):
            # Between-chain variance
            chain_means = np.mean(chains[:, :, j], axis=1)
            overall_mean = np.mean(chain_means)
            B = n_iterations * np.var(chain_means, ddof=1)
            
            # Within-chain variance
            W = np.mean(np.var(chains[:, :, j], axis=1, ddof=1))
            
            # Weighted average of within and between chain variances
            V = (1 - 1/n_iterations) * W + B/n_iterations
            
            # R-hat statistic
            r_hat[j] = np.sqrt(V/W)
            
        return r_hat
    
    def _compute_effective_sample_size(self):
        """
        Compute effective sample size using autocorrelation estimates with improved
        numerical stability and edge case handling.
        """
        try:
            chains = self.results['chains']
            n_chains, n_iterations, n_params = chains.shape
            
            if n_iterations < 2:
                return 0
                
            # Use second half of chains to avoid burn-in effects
            start_idx = n_iterations // 2
            chains = chains[:, start_idx:, :]
            
            # Combine chains
            combined_chain = chains.reshape(-1, n_params)
            
            # Compute ESS for each parameter
            ess_values = []
            
            for j in range(n_params):
                # Extract parameter chain
                param_chain = combined_chain[:, j]
                
                # Skip if chain has no variation
                if np.all(param_chain == param_chain[0]):
                    ess_values.append(1.0)
                    continue
                
                # Compute autocorrelation up to lag 50
                max_lag = min(50, len(param_chain) // 3)
                auto_corr = np.zeros(max_lag)
                
                # Normalize chain
                normalized_chain = (param_chain - np.mean(param_chain)) / np.std(param_chain)
                
                for lag in range(max_lag):
                    if lag >= len(normalized_chain) - 1:
                        break
                    # Direct computation of autocorrelation
                    auto_corr[lag] = np.mean(
                        normalized_chain[lag:] * normalized_chain[:(len(normalized_chain)-lag)]
                    )
                
                # Find first negative or near-zero autocorrelation
                cutoff = np.where(np.abs(auto_corr) < 0.05)[0]
                if len(cutoff) > 0:
                    max_lag = cutoff[0]
                
                # Compute ESS with safeguards
                sum_auto = np.sum(auto_corr[:max_lag])
                if sum_auto >= -1:  # Check for valid autocorrelation sum
                    ess = n_chains * (n_iterations - start_idx) / (1 + 2 * sum_auto)
                    ess_values.append(max(1.0, min(ess, n_chains * (n_iterations - start_idx))))
                else:
                    ess_values.append(1.0)
            
            # Return mean ESS across parameters
            return np.mean(ess_values)
            
        except Exception as e:
            warnings.warn(f"Error computing effective sample size: {str(e)}")
            return 1.0  # Return minimum ESS on error
    
    def _compute_acceptance_rates(self):
        """
        Compute MCMC acceptance rates across all chains.
        """
        if 'acceptance_rates' in self.results:
            return np.mean(self.results['acceptance_rates'])
        return None
    
    def _compute_option_pricing_errors(self):
        """
        Compute option pricing errors across different moneyness levels.
        """
        diagnostics = {}
        
        if 'predicted_option_prices' not in self.results or 'true_option_prices' not in self.results:
            return diagnostics
        
        pred_prices = np.array(self.results['predicted_option_prices'])
        true_prices = np.array(self.results['true_option_prices'])
        
        # Overall RMSE
        rmse = np.sqrt(np.mean((pred_prices - true_prices)**2))
        diagnostics['rmse'] = rmse
        
        # RMSE by moneyness if moneyness data is available
        if 'moneyness' in self.results:
            moneyness = np.array(self.results['moneyness'])
            moneyness_bins = [0.8, 0.9, 1.0, 1.1, 1.2]
            rmse_by_moneyness = []
            sample_sizes = []
            
            for i in range(len(moneyness_bins)-1):
                mask = (moneyness >= moneyness_bins[i]) & (moneyness < moneyness_bins[i+1])
                if np.any(mask):
                    rmse_bin = np.sqrt(np.mean((pred_prices[mask] - true_prices[mask])**2))
                    rmse_by_moneyness.append(rmse_bin)
                    sample_sizes.append(np.sum(mask))
                else:
                    rmse_by_moneyness.append(np.nan)
                    sample_sizes.append(0)
            
            diagnostics['rmse_by_moneyness'] = rmse_by_moneyness
            diagnostics['sample_sizes'] = sample_sizes
        
        return diagnostics
    
    def _compute_parameter_diagnostics(self):
        """
        Compute parameter estimation accuracy metrics.
        """
        diagnostics = {}
        
        # Extract posterior means and true parameters
        posterior_means = self.results['posterior_means']
        true_params = self.results['true_params']
        
        # Compute relative errors
        rel_errors = {}
        for param in true_params:
            if param in posterior_means:
                true_val = true_params[param]
                est_val = posterior_means[param]
                if true_val != 0:
                    rel_errors[param] = abs(est_val - true_val) / abs(true_val)
        
        diagnostics['relative_errors'] = rel_errors
        
        # Compute posterior standard deviations if available
        if 'chains' in self.results:
            chains = self.results['chains']
            posterior_stds = np.std(chains.reshape(-1, chains.shape[2]), axis=0)
            diagnostics['posterior_stds'] = dict(zip(self.results['param_names'], posterior_stds))
        
        return diagnostics