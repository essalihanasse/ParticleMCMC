import numpy as np
class ModelDiagnostics:
    def compute_pricing_errors(self, true_prices, model_prices, by_moneyness=True):
        errors = true_prices - model_prices
        rmse = np.sqrt(np.mean(errors**2))
        
        if by_moneyness:
            moneyness_bins = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
            rmse_by_moneyness = []
            for i in range(len(moneyness_bins)-1):
                mask = (moneyness >= moneyness_bins[i]) & \
                       (moneyness < moneyness_bins[i+1])
                rmse_by_moneyness.append(
                    np.sqrt(np.mean(errors[mask]**2)))
            return rmse, rmse_by_moneyness
        return rmse

    def analyze_convergence(self, chains):
        # Gelman-Rubin diagnostic
        W = np.mean(np.var(chains, axis=1))
        B = np.var(np.mean(chains, axis=1))
        var_theta = (1 - 1/len(chains[0]))*W + B
        R_hat = np.sqrt(var_theta/W)
        return R_hat
    # Add to model_diagnostics.py

    def analyze_filtering_performance(self, true_states, filtered_states):
        """
        Compute comprehensive filtering performance metrics
        """
        # RMSE
        rmse = np.sqrt(np.mean((true_states - filtered_states)**2))
        
        # MAE
        mae = np.mean(np.abs(true_states - filtered_states))
        
        # State correlation
        corr = np.corrcoef(true_states, filtered_states)[0,1]
        
        # Likelihood score (higher is better)
        likelihood = -0.5 * np.sum((true_states - filtered_states)**2 / 
                                np.var(filtered_states))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'correlation': corr,
            'likelihood_score': likelihood
        }

    def compute_particle_filter_diagnostics(self, particles, weights):
        """
        Compute diagnostics for particle filter performance
        """
        # Effective Sample Size
        ess = 1/np.sum(weights**2, axis=0)
        
        # Resampling frequency
        resampling_freq = np.mean(ess < (particles.shape[0]/2))
        
        # Particle diversity (entropy of weight distribution)
        entropy = -np.sum(weights * np.log(weights + 1e-10), axis=0)
        
        return {
            'mean_ess': np.mean(ess),
            'min_ess': np.min(ess),
            'resampling_frequency': resampling_freq,
            'weight_entropy': np.mean(entropy)
        }