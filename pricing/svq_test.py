import numpy as np
from numba import njit, prange
import numpy.random as rd

class SVQMethod:
    def __init__(self, num_quantiles=12, poly_order=3):
        """Initialize SVQ method parameters"""
        self.num_quantiles = num_quantiles
        self.poly_order = poly_order

    def compute_option_weights(self, particles, observations, model):
        """Compute option weights using quantile approach"""
        weights = np.ones(len(particles))

        # Get quantiles of variance distribution  
        quantiles = np.quantile(particles, np.linspace(0, 1, self.num_quantiles))
        
        # Compute option prices at quantile points
        for option in observations:
            # Fit polynomial to price-variance relationship
            prices = []
            for v in quantiles:
                prices.append(self._compute_option_price(v, option, model))
            coeffs = np.polyfit(quantiles, prices, self.poly_order)
            
            # Compute prices for all particles using polynomial
            model_prices = np.polyval(coeffs, particles)
            
            # Update weights based on option fit
            sigma_c = model.params['sigma_c']
            option_weights = np.exp(-0.5 * ((option['price'] - model_prices) / sigma_c)**2) / \
                           (sigma_c * np.sqrt(2 * np.pi))
            weights *= option_weights
            
        return weights

    # def _compute_option_price(self, v, option_params, model):
    #     """Compute single option price using FFT"""
    #     # FFT parameters
    #     alpha = 1.5 
    #     N = 512
    #     B = 500
    #     dx = B/N
    #     dk = 2*np.pi/(N*dx)
        
    #     k = np.arange(N)*dk - B
    #     x = np.arange(N)*dx
        
    #     # Compute characteristic function
    #     cf = self._compute_char_func(k - (alpha+1)*1j, v, option_params, model)
        
    #     # Apply FFT
    #     integrand = np.exp(-1j*k*np.log(option_params['K']))* \
    #                cf/(alpha**2 + alpha - k**2 + 1j*(2*alpha+1)*k)
                   
    #     return np.real(np.exp(-alpha*np.log(option_params['K']))/np.pi * \
    #            np.sum(integrand * np.exp(1j*k*x[0]) * dk))
