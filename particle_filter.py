import numpy as np
from numba import njit, prange
import numpy.random as rd
from scipy.fft import fft

@njit
def systematic_resample(weights):
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

class ParticleFilter:
    def __init__(self, num_particles=50, num_quantiles=12):
        self.num_particles = num_particles
        self.num_quantiles = num_quantiles
        
    def initialize_particles(self, initial_state, noise_scale=0.1):
        """Initialize particles with normal noise around initial state"""
        return np.random.normal(initial_state, noise_scale, self.num_particles)

    def compute_likelihood(self, observations, params):
        """Compute log-likelihood using particle filter with quantile-based approach"""
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
        log_likelihood = 0.0
        
        # Main filtering loop
        for t in range(1, len(prices)):
            # Compute log-return
            log_return = np.log(prices[t] / prices[t-1])
            
            # Propagate particles
            particles = self._propagate_particles(particles, prop_params, dt)
            
            # Compute weights using returns and options
            weights = self._compute_return_weights(log_return, particles, ret_params, dt)
            
            # Add option weights if available using quantile method
            if len(options[t]) > 0:
                option_weights = self._compute_option_weights_quantile(options[t], particles, params)
                weights *= option_weights
            
            # Normalize weights
            weights = np.maximum(weights, 1e-300)  
            weights /= np.sum(weights)
            
            # Update likelihood
            log_likelihood += np.log(np.mean(weights))
            
            # Resample if effective sample size is too low
            ESS = 1 / np.sum(weights**2)
            if ESS < self.num_particles/2:
                indices = systematic_resample(weights)
                particles = particles[indices]
                weights = np.ones(self.num_particles) / self.num_particles
        
        return log_likelihood

    def _propagate_particles(self, particles, param_array, dt):
        """Propagate particles forward using transition dynamics"""
        kappa, theta, sigma, lmda, mu_v = param_array
        new_particles = np.empty_like(particles)
        
        for i in prange(len(particles)):
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
        """Compute option weights using quantile-based approach"""
        weights = np.ones(self.num_particles)
        
        # Get quantiles of variance distribution
        quantiles = np.quantile(particles, np.linspace(0, 1, self.num_quantiles))
        
        for option in options:
            # Compute option prices at quantile points
            quant_prices = []
            for v in quantiles:
                price = self._compute_option_price(v, option, params)
                quant_prices.append(price)
            
            # Fit polynomial to price-variance relationship
            coeffs = np.polyfit(quantiles, quant_prices, 3)
            
            # Compute prices for all particles
            model_prices = np.polyval(coeffs, particles)
            
            # Update weights based on model fit
            sigma_c = params['sigma_c']
            option_weights = np.exp(-0.5 * ((option['price'] - model_prices) / sigma_c)**2) / \
                           (sigma_c * np.sqrt(2 * np.pi))
            weights *= option_weights
            
        return weights

    def _compute_option_price(self, V, option, params):
        """Compute option price using FFT method"""
        # FFT parameters
        N = 512 
        alpha = 1.5
        eta = 0.25
        lambda_ = 2 * np.pi / (N * eta)
        b = np.pi / eta
        
        v = np.arange(N) * eta
        k = -b + lambda_ * np.arange(N)
        
        # Characteristic function computation
        u = v - (alpha + 1)*1j
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
        
        cf = np.exp(C + D*V)
        
        # Modified characteristic function
        psi = np.exp(-option['r'] * option['tau']) * cf / \
              (alpha**2 + alpha - v**2 + 1j*(2*alpha + 1)*v)
        
        # FFT
        x = np.exp(1j * b * v) * psi * eta
        fft_result = np.real(fft(x))
        
        # Price interpolation
        log_strike = np.log(option['K'])
        idx = int((log_strike + b)/lambda_)
        if 0 <= idx < N:
            price = np.exp(-alpha * log_strike) / np.pi * fft_result[idx]
            return max(price, 0)
        return 0.0