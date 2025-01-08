import numpy as np
from numba import njit, prange
import numpy.random as rd

class ParticleFilter:
    def __init__(self, num_particles=50, num_quantiles=12):
        self.num_particles = num_particles
        self.num_quantiles = num_quantiles
        self.filtered_states = None
        self.filtered_std = None
        self.log_likelihood = None
        self.state_history = None
        
    def initialize_particles(self, initial_state, data, params):
        """Improved particle initialization using first few observations."""
        # Use first few observations to estimate initial variance range
        V0_est = self._estimate_initial_variance(data[:min(10, len(data))], params)
        base_particles = np.random.normal(initial_state, V0_est * 0.1, self.num_particles)
        return np.maximum(base_particles, 1e-7)
    def _estimate_initial_variance(self, initial_data, params):
        """Estimate initial variance using method of moments on returns."""
        returns = np.diff(np.log(initial_data))
        var_est = np.var(returns)
        # Adjust for jump component if present
        if 'lambda' in params:
            var_est -= params['lambda'] * (params['mu_s']**2 + params['sigma_s']**2)
        return max(var_est, 1e-5)
    
    def compute_likelihood(self, observations, params):
        """Compute log-likelihood with improved numerical stability."""
        try:
            prices = observations['prices']
            options = observations['options']
            dt = 1/252
            
            # Convert parameters to arrays for efficiency
            prop_params = np.array([
                params['kappa'], 
                params['theta'], 
                params['sigma'], 
                params['lmda'] if 'lmda' in params else 0, 
                params['mu_v'] if 'mu_v' in params else 0
            ])
            
            ret_params = np.array([
                params['r'], 
                params['delta'], 
                params['eta_s']
            ])
            
            # Initialize particles with validation
            particles = self.initialize_particles(params['V0'], prices, params)
            if particles is None or len(particles) == 0:
                raise ValueError("Failed to initialize particles")
            
            n_timesteps = len(prices)
            log_likelihood = 0.0
            filtered_states = np.zeros(n_timesteps)
            filtered_std = np.zeros(n_timesteps)
            
            # Store initial state
            filtered_states[0] = np.mean(particles)
            filtered_std[0] = np.std(particles)
            
            # Main filtering loop with improved numerical stability
            for t in range(1, n_timesteps):
                # Compute log-return with validation
                log_return = np.log(prices[t] / prices[t-1])
                if not np.isfinite(log_return):
                    print(f"Warning: Invalid log return at time {t}")
                    continue
                
                # Propagate particles
                new_particles = self._propagate_particles(particles, prop_params, dt)
                particles = new_particles
                
                # Compute log weights
                log_weights = self._compute_log_weights(log_return, particles, ret_params, dt)
                
                # Add option weights if available
                if len(options[t]) > 0:
                    option_log_weights = self._compute_option_log_weights_quantile(
                        options[t], particles, params
                    )
                    log_weights += option_log_weights
                
                # Normalize weights with improved stability
                max_log_weight = np.max(log_weights)
                weights = np.exp(log_weights - max_log_weight)
                sum_weights = np.sum(weights)
                
                if sum_weights > 0:
                    weights /= sum_weights
                    log_likelihood += max_log_weight + np.log(sum_weights) - np.log(self.num_particles)
                else:
                    print(f"Warning: Zero sum of weights at time {t}")
                    weights = np.ones_like(weights) / len(weights)
                
                # Update state estimates
                filtered_states[t] = np.sum(weights * particles)
                filtered_std[t] = np.sqrt(np.sum(weights * (particles - filtered_states[t])**2))
                
                # Systematic resampling when effective sample size is too low
                ESS = 1 / np.sum(weights**2)
                if ESS < self.num_particles/2:
                    indices = self._systematic_resample(weights)
                    particles = particles[indices]
            
            self.filtered_states = filtered_states
            self.filtered_std = filtered_std
            self.log_likelihood = log_likelihood
            
            return log_likelihood

        except Exception as e:
            print(f"Error in compute_likelihood: {str(e)}")
            return -np.inf

    def _propagate_particles(self, particles, param_array, dt):
        """Propagate particles forward"""
        kappa, theta, sigma, lmda, mu_v = param_array
        new_particles = np.empty_like(particles)
        
        for i in range(len(particles)):
            drift = kappa * (theta - particles[i]) * dt
            diffusion = sigma * np.sqrt(max(particles[i], 1e-7) * dt) * rd.normal()
            jump = 0.0
            if rd.random() < lmda * dt:
                jump = rd.exponential(mu_v)
            new_particles[i] = max(particles[i] + drift + diffusion + jump, 1e-7)
            
        return new_particles

    def _compute_return_weights(self, log_return, particles, param_array, dt):
        """Compute weights from return likelihood"""
        r, delta, eta_s = param_array
        mean = (r - delta - particles/2 + eta_s * particles) * dt
        std = np.sqrt(np.maximum(particles * dt, 1e-7))
        weights = np.exp(-0.5 * ((log_return - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
        return weights
    def _compute_log_weights(self, log_return, particles, param_array, dt):
        """Compute weights in log space for numerical stability."""
        r, delta, eta_s = param_array
        mean = (r - delta - particles/2 + eta_s * particles) * dt
        var = np.maximum(particles * dt, 1e-7)
        log_weights = -0.5 * ((log_return - mean)**2 / var) - 0.5 * np.log(2 * np.pi * var)
        return log_weights
    def _compute_option_weights_quantile(self, options, particles, params):
        """Compute option weights using quantile method"""
        weights = np.ones(self.num_particles)
        
        quantiles = np.quantile(particles, np.linspace(0, 1, self.num_quantiles))
        
        for option in options:
            prices = []
            for v in quantiles:
                prices.append(self._compute_option_price(v, option, params))
            coeffs = np.polyfit(quantiles, prices, 3)
            
            model_prices = np.polyval(coeffs, particles)
            
            sigma_c = params['sigma_c']
            option_weights = np.exp(-0.5 * ((option['price'] - model_prices) / sigma_c)**2) / \
                           (sigma_c * np.sqrt(2 * np.pi))
            weights *= option_weights
            
        return weights

    def _systematic_resample(self, weights):
        """Improved systematic resampling implementation."""
        N = len(weights)
        # Generate random starting point
        u = np.random.uniform(0, 1/N)
        # Generate equally spaced points
        positions = (u + np.arange(N)) / N
        # Generate cumulative sum
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0  # Avoid numerical errors
        
        # Find indices
        indices = np.zeros(N, dtype=np.int64)
        i = 0
        for j in range(N):
            while cumsum[i] < positions[j]:
                i += 1
            indices[j] = i
        
        return indices

    def _compute_option_price(self, v, option, params):
        """Compute option price using FFT"""
        # FFT parameters
        N = 512
        alpha = 1.5
        eta = 0.25
        lambda_ = 2 * np.pi / (N * eta)
        b = np.pi / eta
        
        v_grid = np.arange(N) * eta
        k = -b + lambda_ * np.arange(N)
        
        u = v_grid - (alpha + 1)*1j
        kappa_Q = params['kappa'] - params['eta_v']
        theta_Q = params['kappa'] * params['theta'] / kappa_Q
        
        d = np.sqrt((params['sigma']**2)*(u**2 + u*1j) + 
                   (kappa_Q - params['rho']*params['sigma']*u*1j)**2)
        
        g = (kappa_Q - params['rho']*params['sigma']*u*1j - d) / \
            (kappa_Q - params['rho']*params['sigma']*u*1j + d)
        
        C = (kappa_Q * theta_Q / params['sigma']**2) * \
            ((kappa_Q - params['rho']*params['sigma']*u*1j - d)*option['tau'] - 
             2*np.log((1 - g*np.exp(-d*option['tau']))/(1 - g)))
        
        D = (kappa_Q - params['rho']*params['sigma']*u*1j - d) * \
            (1 - np.exp(-d*option['tau']))/(1 - g*np.exp(-d*option['tau'])) / \
            params['sigma']**2
        
        cf = np.exp(C + D*v)
        
        psi = np.exp(-option['r'] * option['tau']) * cf / \
              (alpha**2 + alpha - v_grid**2 + 1j*(2*alpha + 1)*v_grid)
        
        x = np.exp(1j * b * v_grid) * psi * eta
        fft_result = np.real(np.fft.fft(x))
        
        log_strike = np.log(option['K'])
        idx = int((log_strike + b)/lambda_)
        if 0 <= idx < N:
            price = np.exp(-alpha * log_strike) / np.pi * fft_result[idx]
            return max(price, 0)
        return 0.0
    # Add this method to your ParticleFilter class
    def _compute_option_log_weights_quantile(self, options, particles, params):
        """
        Compute option weights using quantile method in log space for numerical stability.
        
        Args:
            options (list): List of option dictionaries
            particles (ndarray): Array of particle states
            params (dict): Model parameters
            
        Returns:
            ndarray: Log weights for each particle
        """
        log_weights = np.zeros(self.num_particles)
        
        # Compute quantiles of particles
        quantiles = np.quantile(particles, np.linspace(0, 1, self.num_quantiles))
        
        for option in options:
            # Calculate option prices for quantiles
            quantile_prices = []
            for v in quantiles:
                price = self._compute_option_price(v, option, params)
                quantile_prices.append(price)
                
            # Fit polynomial to quantile prices
            coeffs = np.polyfit(quantiles, quantile_prices, 3)
            
            # Compute model prices for all particles
            model_prices = np.polyval(coeffs, particles)
            
            # Compute log weights using normal likelihood
            sigma_c = params['sigma_c']
            particle_log_weights = (-0.5 * ((option['price'] - model_prices) / sigma_c)**2 
                                - np.log(sigma_c) - 0.5 * np.log(2 * np.pi))
            
            # Add to total log weights
            log_weights += particle_log_weights
        
        return log_weights