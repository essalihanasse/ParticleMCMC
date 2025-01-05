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
        
    def initialize_particles(self, initial_state, noise_scale=0.1):
        """Initialize particles with normal noise around initial state"""
        return np.random.normal(initial_state, noise_scale, self.num_particles)

    def compute_likelihood(self, observations, params):
        """
        Compute log-likelihood using particle filter with quantile-based approach
        
        Parameters:
        -----------
        observations : dict
            Dictionary containing 'prices' and 'options'
        params : dict
            Model parameters
        
        Returns:
        --------
        float
            Log-likelihood value
        """
        prices = observations['prices']
        options = observations['options']
        dt = 1/252
        
        # Convert parameter dictionary to arrays for Numba functions
        prop_params = np.array([
            params['kappa'], 
            params['theta'], 
            params['sigma'], 
            params['lmda'], 
            params['mu_v']
        ])
        
        ret_params = np.array([
            params['r'], 
            params['delta'], 
            params['eta_s']
        ])
        
        # Initialize particles
        particles = self.initialize_particles(params['V0'])
        particles = np.maximum(particles, 1e-7)
        
        # Initialize storage arrays
        log_likelihood = 0.0
        n_timesteps = len(prices)
        filtered_states = np.zeros(n_timesteps)
        filtered_std = np.zeros(n_timesteps)
        
        # Store initial state
        filtered_states[0] = np.mean(particles)
        filtered_std[0] = np.std(particles)
        
        # Main filtering loop
        for t in range(1, n_timesteps):
            # Compute log-return
            log_return = np.log(prices[t] / prices[t-1])
            
            # Propagate particles
            new_particles = self._propagate_particles(particles, prop_params, dt)
            particles = new_particles
            
            # Compute weights using returns
            weights = self._compute_return_weights(log_return, particles, ret_params, dt)
            
            # Add option weights if available
            if len(options[t]) > 0:
                option_weights = self._compute_option_weights_quantile(
                    options[t], particles, params
                )
                weights *= option_weights
            
            # Normalize weights
            weights = np.maximum(weights, 1e-300)
            weights /= np.sum(weights)
            
            # Update likelihood
            log_likelihood += np.log(np.mean(weights))
            
            # Store filtered states
            filtered_states[t] = np.sum(weights * particles)
            filtered_std[t] = np.sqrt(
                np.sum(weights * (particles - filtered_states[t])**2)
            )
            
            # Resample if needed
            ESS = 1 / np.sum(weights**2)
            if ESS < self.num_particles/2:
                indices = self._systematic_resample(weights)
                particles = particles[indices]
                weights = np.ones(self.num_particles) / self.num_particles
        
        # Store results as class attributes
        self.filtered_states = filtered_states
        self.filtered_std = filtered_std
        self.log_likelihood = log_likelihood
        
        return log_likelihood

    def _propagate_particles(self, particles, param_array, dt):
        """Propagate particles forward using transition dynamics"""
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
        """Compute weights based on return likelihood"""
        r, delta, eta_s = param_array
        mean = (r - delta - particles/2 + eta_s * particles) * dt
        std = np.sqrt(np.maximum(particles * dt, 1e-7))
        weights = np.exp(-0.5 * ((log_return - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
        return weights

    def _compute_option_weights_quantile(self, options, particles, params):
        """Compute option weights using quantile method"""
        weights = np.ones(self.num_particles)
        
        # Get quantiles of variance distribution
        quantiles = np.quantile(particles, np.linspace(0, 1, self.num_quantiles))
        
        for option in options:
            # Fit polynomial to prices at quantile points
            prices = []
            for v in quantiles:
                prices.append(self._compute_option_price(v, option, params))
            coeffs = np.polyfit(quantiles, prices, 3)
            
            # Compute prices for all particles
            model_prices = np.polyval(coeffs, particles)
            
            # Update weights
            sigma_c = params['sigma_c']
            option_weights = np.exp(-0.5 * ((option['price'] - model_prices) / sigma_c)**2) / \
                           (sigma_c * np.sqrt(2 * np.pi))
            weights *= option_weights
            
        return weights

    def _systematic_resample(self, weights):
        """Perform systematic resampling"""
        N = len(weights)
        positions = (rd.random() + np.arange(N)) / N
        indices = np.zeros(N, dtype=np.int64)
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        
        i = 0
        for j in range(N):
            while cumsum[i] < positions[j]:
                i += 1
            indices[j] = i
            
        return indices

    def _compute_option_price(self, v, option, params):
        """Compute option price using FFT method"""
        # FFT parameters
        N = 512
        alpha = 1.5
        eta = 0.25
        lambda_ = 2 * np.pi / (N * eta)
        b = np.pi / eta
        
        v_grid = np.arange(N) * eta
        k = -b + lambda_ * np.arange(N)
        
        # Characteristic function computation
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