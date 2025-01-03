import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class OptimizedPMMH:
    def __init__(self, num_iterations=50, num_chains=5, num_vertical=10, num_horizontal=1, use_orthogonal=True):
        self.num_iterations = num_iterations
        self.num_chains = num_chains
        self.num_vertical = num_vertical
        self.num_horizontal = num_horizontal
        self.use_orthogonal = use_orthogonal
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)

    def run(self, observations, particle_filter, initial_params, progress_callback=None):
        if self.use_orthogonal:
            return self._run_orthogonal(observations, particle_filter, initial_params, progress_callback)
        else:
            return self._run_adaptive(observations, particle_filter, initial_params, progress_callback)

    def _run_orthogonal(self, observations, particle_filter, initial_params, progress_callback=None):
        """Run orthogonal MCMC with horizontal and vertical moves"""
        param_names = list(initial_params.keys())
        
        # Initialize chains
        chains = []
        current_params = []
        current_lls = []
        
        for _ in range(self.num_chains):
            chains.append([])
            current_params.append(initial_params.copy())
            current_lls.append(particle_filter.compute_likelihood(observations, initial_params))
            
        accepted = np.zeros(self.num_chains)
        
        # Run parallel chains with vertical and horizontal moves
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for iter in range(self.num_iterations):
                # Vertical moves (MH step within each chain)
                if iter % self.num_vertical == 0:
                    futures = []
                    for chain in range(self.num_chains):
                        futures.append(
                            executor.submit(
                                self._vertical_move,
                                observations,
                                particle_filter,
                                current_params[chain],
                                current_lls[chain],
                                iter
                            )
                        )
                    
                    for chain, future in enumerate(futures):
                        new_params, new_ll, was_accepted = future.result()
                        current_params[chain] = new_params
                        current_lls[chain] = new_ll
                        chains[chain].append(list(new_params.values()))
                        accepted[chain] += was_accepted
                        
                        if progress_callback:
                            progress_callback(1)
                
                # Horizontal moves (state exchange between chains)
                if iter % (self.num_vertical * self.num_horizontal) == 0:
                    for i in range(self.num_chains - 1):
                        log_ratio = current_lls[i+1] - current_lls[i]
                        if np.log(np.random.random()) < log_ratio:
                            # Exchange states
                            current_params[i], current_params[i+1] = \
                                current_params[i+1], current_params[i]
                            current_lls[i], current_lls[i+1] = \
                                current_lls[i+1], current_lls[i]
        
        # Calculate acceptance rates
        acceptance_rates = accepted / self.num_iterations
        chains = np.array([np.array(chain) for chain in chains])
            
        return chains, acceptance_rates

    def _run_adaptive(self, observations, particle_filter, initial_params, progress_callback=None):
        """Run standard adaptive MCMC"""
        param_names = list(initial_params.keys())
        chain = []
        current_params = initial_params.copy()
        current_ll = particle_filter.compute_likelihood(observations, current_params)
        
        # Initialize adaptive covariance
        param_vec = list(current_params.values())
        cov = np.diag(np.square(np.array(param_vec) * 0.01))
        
        accepted = 0
        for iter in range(self.num_iterations):
            # Update proposal covariance after burnin
            if iter > 100:
                sample_cov = np.cov(np.array(chain).T)
                if not np.any(np.isnan(sample_cov)):
                    cov = 2.4**2 / len(param_vec) * sample_cov + 1e-6 * np.eye(len(param_vec))
            
            # Propose new parameters
            proposed_vec = np.random.multivariate_normal(list(current_params.values()), cov)
            proposed_params = dict(zip(current_params.keys(), proposed_vec))
            
            # Check constraints
            if self._check_constraints(proposed_params):
                proposed_ll = particle_filter.compute_likelihood(observations, proposed_params)
                
                # Accept/reject
                log_alpha = proposed_ll - current_ll
                if np.log(np.random.random()) < log_alpha:
                    current_params = proposed_params.copy()
                    current_ll = proposed_ll
                    accepted += 1
            
            chain.append(list(current_params.values()))
            
            if progress_callback:
                progress_callback(1)
                
        acceptance_rate = accepted / self.num_iterations
        return np.array([chain]), np.array([acceptance_rate])

    def _vertical_move(self, observations, particle_filter, current_params, current_ll, iter):
        """Execute single vertical MCMC move"""
        # Propose parameters
        proposed_params = self._propose_parameters(current_params, iter)
        
        if not self._check_constraints(proposed_params):
            return current_params, current_ll, False
            
        proposed_ll = particle_filter.compute_likelihood(observations, proposed_params)
        
        # Accept/reject
        log_alpha = proposed_ll - current_ll
        if np.log(np.random.random()) < log_alpha:
            return proposed_params, proposed_ll, True
            
        return current_params, current_ll, False

    # In PMMH.py

    def _propose_parameters(self, current_params, iter=0):
        """Propose new parameters using adaptive scales with strict bounds"""
        scales = {
            'kappa': 0.1,
            'theta': 0.002,
            'sigma': 0.02,
            'rho': 0.05,
            'eta_s': 0.1,
            'eta_v': 0.05,
            'lmda': 0.05,
            'mu_s': 0.005,
            'sigma_s': 0.01,
            'eta_js': 0.005,
            'mu_v': 0.005,
            'eta_jv': 0.005,
            'rho_j': 0.05,
            'sigma_c': 0.1,
            'V0': 0.002,
            'r': 0.001,
            'delta': 0.001
        }
        
        proposed = {}
        for param, value in current_params.items():
            scale = scales.get(param, 0.1)
            if iter > 100:  # Adapt scales after burnin
                scale *= 2.4 / np.sqrt(len(current_params))
            
            # Handle parameters that must be positive
            if param in ['kappa', 'theta', 'sigma', 'eta_s', 'eta_v', 'lmda', 
                        'sigma_s', 'sigma_c', 'mu_v', 'eta_jv', 'V0']:
                # Use log-normal proposal for positive parameters
                log_value = np.log(value)
                proposed_log = log_value + np.random.normal(0, scale)
                proposed[param] = np.exp(proposed_log)
            # Handle correlation parameters
            elif param in ['rho', 'rho_j']:
                # Use logit transform for parameters bounded in (-1, 1)
                x = 0.5 * (value + 1)  # Transform from (-1,1) to (0,1)
                x = np.clip(x, 0.001, 0.999)  # Avoid boundary issues
                logit_x = np.log(x / (1 - x))
                proposed_logit = logit_x + np.random.normal(0, scale)
                proposed_x = 1 / (1 + np.exp(-proposed_logit))
                proposed[param] = 2 * proposed_x - 1  # Transform back to (-1,1)
            else:
                # Standard normal proposal for unbounded parameters
                proposed[param] = value + np.random.normal(0, scale)
        
        return proposed

    def _check_constraints(self, params):
        """Check parameter constraints with numerical stability"""
        constraints = [
            params['kappa'] > 1e-7,
            params['theta'] > 1e-7,
            params['sigma'] > 1e-7,
            np.abs(params['rho']) < 0.9999,
            params['lmda'] > 1e-7,
            params['sigma_s'] > 1e-7,
            params['sigma_c'] > 1e-7,
            np.abs(params.get('rho_j', 0)) < 0.9999,
            params.get('V0', 1e-4) > 1e-7,
            params.get('r', 0) >= 0,
            params.get('delta', 0) >= 0,
            params.get('mu_v', 0.01) > 1e-7,  # Ensure positive mean for exponential
            params.get('eta_jv', 0.01) > 1e-7  # Ensure positive scale for exponential
        ]
        return all(constraints)