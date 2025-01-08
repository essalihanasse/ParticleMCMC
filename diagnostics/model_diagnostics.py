import numpy as np

class ModelDiagnostics:
    def compute_pricing_errors(self, true_prices, model_prices, moneyness=None, by_moneyness=True):
        """
        Compute pricing errors with optional moneyness-based breakdown
        
        Parameters:
        -----------
        true_prices: np.ndarray
            Observed market prices
        model_prices: np.ndarray
            Model-implied prices
        moneyness: np.ndarray, optional
            Moneyness values for each price
        by_moneyness: bool
            Whether to compute errors by moneyness buckets
        
        Returns:
        --------
        tuple: (overall RMSE, RMSE by moneyness) if by_moneyness=True
               float: overall RMSE if by_moneyness=False
        """
        if not isinstance(true_prices, np.ndarray):
            true_prices = np.array(true_prices)
        if not isinstance(model_prices, np.ndarray):
            model_prices = np.array(model_prices)
            
        if true_prices.shape != model_prices.shape:
            raise ValueError("True and model prices must have same shape")
            
        errors = true_prices - model_prices
        rmse = np.sqrt(np.mean(np.square(errors)))
        
        if by_moneyness:
            if moneyness is None:
                raise ValueError("Moneyness values required for moneyness-based analysis")
                
            moneyness_bins = np.array([0.9, 0.95, 1.0, 1.05, 1.1])
            rmse_by_moneyness = []
            
            for i in range(len(moneyness_bins)-1):
                mask = (moneyness >= moneyness_bins[i]) & \
                       (moneyness < moneyness_bins[i+1])
                if np.any(mask):
                    bin_rmse = np.sqrt(np.mean(np.square(errors[mask])))
                    rmse_by_moneyness.append(bin_rmse)
                else:
                    rmse_by_moneyness.append(np.nan)
                    
            return rmse, rmse_by_moneyness
        
        return rmse

    def analyze_convergence(self, chains):
        """
        Compute Gelman-Rubin convergence diagnostic
        
        Parameters:
        -----------
        chains: np.ndarray
            MCMC chains with shape (n_chains, n_samples, n_params)
            
        Returns:
        --------
        np.ndarray: R-hat statistics for each parameter
        """
        if not isinstance(chains, np.ndarray):
            chains = np.array(chains)
            
        if len(chains.shape) != 3:
            raise ValueError("Chains should have shape (n_chains, n_samples, n_params)")
            
        n_chains, n_samples, n_params = chains.shape
        
        R_hats = np.zeros(n_params)
        for p in range(n_params):
            chain_means = np.mean(chains[:, :, p], axis=1)
            chain_vars = np.var(chains[:, :, p], axis=1, ddof=1)
            
            W = np.mean(chain_vars)  # Within-chain variance
            B = np.var(chain_means, ddof=1)  # Between-chain variance
            
            # Compute variance estimator
            var_theta = ((n_samples - 1)/n_samples) * W + B
            
            # Compute R-hat
            R_hats[p] = np.sqrt(var_theta/W) if W > 0 else np.inf
            
        return R_hats

    def analyze_filtering_performance(self, true_states, filtered_states):
        """
        Compute comprehensive filtering performance metrics
        
        Parameters:
        -----------
        true_states: np.ndarray
            True state values
        filtered_states: np.ndarray
            Filtered state estimates
            
        Returns:
        --------
        dict: Dictionary of performance metrics
        """
        if not isinstance(true_states, np.ndarray):
            true_states = np.array(true_states)
        if not isinstance(filtered_states, np.ndarray):
            filtered_states = np.array(filtered_states)
            
        if true_states.shape != filtered_states.shape:
            raise ValueError("True and filtered states must have same shape")
            
        # RMSE
        rmse = np.sqrt(np.mean(np.square(true_states - filtered_states)))
        
        # MAE
        mae = np.mean(np.abs(true_states - filtered_states))
        
        # State correlation
        corr = np.corrcoef(true_states.flatten(), filtered_states.flatten())[0,1]
        
        # Log likelihood score
        var_states = np.var(filtered_states)
        if var_states > 0:
            likelihood = -0.5 * np.sum(np.square(true_states - filtered_states) / var_states)
        else:
            likelihood = -np.inf
            
        return {
            'rmse': rmse,
            'mae': mae,
            'correlation': corr,
            'likelihood_score': likelihood
        }

    def compute_particle_filter_diagnostics(self, particles, weights):
        """
        Compute diagnostics for particle filter performance
        
        Parameters:
        -----------
        particles: np.ndarray
            Particle values with shape (n_particles, n_timesteps)
        weights: np.ndarray
            Particle weights with shape (n_particles, n_timesteps)
            
        Returns:
        --------
        dict: Dictionary of particle filter diagnostics
        """
        if not isinstance(particles, np.ndarray):
            particles = np.array(particles)
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
            
        if particles.shape[0] != weights.shape[0]:
            raise ValueError("Particles and weights must have same first dimension")
            
        # Normalize weights if they aren't already
        weights = weights / np.sum(weights, axis=0, keepdims=True)
        
        # Effective Sample Size
        ess = 1/np.sum(np.square(weights), axis=0)
        
        # Resampling threshold
        threshold = particles.shape[0]/2
        resampling_freq = np.mean(ess < threshold)
        
        # Weight entropy (particle diversity measure)
        with np.errstate(divide='ignore'):
            entropy = -np.sum(weights * np.log(weights + 1e-10), axis=0)
        
        # Coefficient of variation of weights
        cv_weights = np.std(weights, axis=0) / np.mean(weights, axis=0)
        
        return {
            'mean_ess': np.mean(ess),
            'min_ess': np.min(ess),
            'resampling_frequency': resampling_freq,
            'weight_entropy': np.mean(entropy),
            'cv_weights': np.mean(cv_weights)
        }