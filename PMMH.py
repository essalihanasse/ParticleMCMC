import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class OptimizedPMMH:
    def __init__(self, num_iterations=50, num_chains=5):
        self.num_iterations = num_iterations
        self.num_chains = num_chains
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)

    def run(self, observations, particle_filter, initial_params, progress_callback=None):
        param_names = list(initial_params.keys())
        
        # Run chains in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for _ in range(self.num_chains):
                futures.append(
                    executor.submit(
                        self._run_single_chain,
                        observations,
                        particle_filter,
                        initial_params.copy(),
                        self.num_iterations
                    )
                )
            
            # Collect results
            chains = []
            acceptance_rates = []
            for future in futures:
                chain, acc_rate = future.result()
                chains.append(chain)
                acceptance_rates.append(acc_rate)
                if progress_callback:
                    progress_callback(1)
        
        return np.array(chains), np.array(acceptance_rates)

    @staticmethod
    def _run_single_chain(observations, particle_filter, params, num_iterations):
        """Run a single MCMC chain with adaptive proposals"""
        chain = np.zeros((num_iterations, len(params)))
        current_params = params.copy()
        
        # Initial likelihood
        current_ll = particle_filter.compute_likelihood(observations, current_params)
        accepted = 0
        
        # Initialize adaptive covariance matrix
        param_vec = list(current_params.values())
        cov = np.diag(np.square(np.array(param_vec) * 0.01))  # Initial scaling
        
        for iter in range(num_iterations):
            # Update proposal covariance using previous iterations
            if iter > 0:
                sample_cov = np.cov(chain[:iter].T)
                if not np.any(np.isnan(sample_cov)):
                    cov = 2.4**2 / len(param_vec) * sample_cov + 1e-6 * np.eye(len(param_vec))
            
            # Propose new parameters using multivariate normal
            proposed_vec = np.random.multivariate_normal(list(current_params.values()), cov)
            proposed_params = dict(zip(current_params.keys(), proposed_vec))
            
            # Check constraints and compute likelihood
            if OptimizedPMMH._check_constraints(proposed_params):
                proposed_ll = particle_filter.compute_likelihood(observations, proposed_params)
                
                # Accept/reject
                log_alpha = proposed_ll - current_ll
                if np.log(np.random.random()) < log_alpha:
                    current_params = proposed_params.copy() 
                    current_ll = proposed_ll
                    accepted += 1
                    
            # Store current parameters
            chain[iter] = list(current_params.values())
            
        acceptance_rate = accepted / num_iterations
        return chain, acceptance_rate
    @staticmethod
    def _propose_parameters(current_params):
        """Propose new parameters using adaptive scales"""
        scales = {
            'kappa': 0.2,
            'theta': 0.005,
            'sigma': 0.05,
            'rho': 0.1,
            'eta_s': 0.3,
            'eta_v': 0.1,
            'lmda': 0.1,
            'mu_s': 0.01,
            'sigma_s': 0.02,
            'eta_js': 0.01,
            'mu_v': 0.01,
            'eta_jv': 0.01,
            'rho_j': 0.1,
            'sigma_c': 0.3,
            'V0': 0.005,
            'r': 0.001,
            'delta': 0.001
        }
        
        proposed = {}
        for param, value in current_params.items():
            scale = scales.get(param, 0.1)
            proposed[param] = value + np.random.normal(0, scale)
        
        return proposed

    @staticmethod
    def _check_constraints(params):
        """Check parameter constraints"""
        constraints = [
            params['kappa'] > 0,
            params['theta'] > 0,
            params['sigma'] > 0,
            -1 < params['rho'] < 1,
            params['lmda'] > 0,
            params['sigma_s'] > 0,
            params['sigma_c'] > 0,
            -1 < params['rho_j'] < 1,
            params.get('V0', 1e-4) > 0,
            params.get('r', 0) >= 0,
            params.get('delta', 0) >= 0
        ]
        return all(constraints)