import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class OptimizedPMMH:
    def __init__(self, num_iterations=50, num_chains=5, num_vertical=10, 
                 num_horizontal=1, use_orthogonal=True, burnin=0):
        self.num_iterations = num_iterations
        self.num_chains = num_chains
        self.num_vertical = num_vertical
        self.num_horizontal = num_horizontal
        self.use_orthogonal = use_orthogonal
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        self.burnin = burnin

    def run(self, observations, particle_filter, initial_params, progress_callback=None):
        """Main run method that chooses between orthogonal and adaptive MCMC"""
        if self.use_orthogonal:
            return self._run_orthogonal(observations, particle_filter, initial_params, progress_callback)
        else:
            return self._run_adaptive(observations, particle_filter, initial_params, progress_callback)

    def _run_orthogonal(self, observations, particle_filter, initial_params, progress_callback=None):
        """Run orthogonal MCMC with horizontal and vertical moves"""
        # Initialize chains
        chains = []
        current_params = []
        current_lls = []
        
        # Initialize each chain
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

    def _vertical_move(self, observations, particle_filter, current_params, current_ll, iter):
        """Execute single vertical MCMC move"""
        # Propose parameters
        proposed_params = self._propose_parameters(current_params, iter)
        
        if not self._check_constraints(proposed_params):
            return current_params, current_ll, False
        
        proposed_ll = particle_filter.compute_likelihood(observations, proposed_params)
        
        # Accept/reject
        log_alpha = float(proposed_ll - current_ll)  # Ensure float type
        if np.log(np.random.random()) < log_alpha:
            return proposed_params, proposed_ll, True
            
        return current_params, current_ll, False

    def _propose_parameters(self, current_params, iter=0):
        """Propose new parameters using adaptive scales"""
        base_scales = {
            'kappa': 0.05,
            'theta': 0.001,
            'sigma': 0.01,
            'rho': 0.02,
            'eta_s': 0.05,
            'eta_v': 0.02,
            'lmda': 0.02,
            'mu_s': 0.002,
            'sigma_s': 0.005,
            'eta_js': 0.002,
            'mu_v': 0.002,
            'eta_jv': 0.002, 
            'rho_j': 0.02,
            'sigma_c': 0.05,
            'V0': 0.001,
            'r': 0.0005,
            'delta': 0.0005
        }
        
        proposed = {}
        for param, value in current_params.items():
            scale = base_scales.get(param, 0.05)
            
            # Adapt scales after burn-in
            if iter > self.burnin:
                scale *= 2.4 / np.sqrt(len(current_params))
            
            if param in ['kappa', 'theta', 'sigma', 'eta_s', 'eta_v', 'lmda', 
                        'sigma_s', 'sigma_c', 'mu_v', 'eta_jv', 'V0']:
                # Use log-normal proposal for positive parameters
                log_value = np.log(value)
                proposed_log = log_value + np.random.normal(0, scale)
                proposed[param] = np.exp(proposed_log)
                
            elif param in ['rho', 'rho_j']:
                # Use logit transform for correlation parameters
                x = 0.5 * (value + 1)
                x = np.clip(x, 0.001, 0.999)
                logit_x = np.log(x / (1 - x))
                proposed_logit = logit_x + np.random.normal(0, scale)
                proposed_x = 1 / (1 + np.exp(-proposed_logit))
                proposed[param] = 2 * proposed_x - 1
                
            else:
                # Normal proposal for unbounded parameters
                proposed[param] = value + np.random.normal(0, scale)
        
        return proposed

    def _check_constraints(self, params):
        """Check parameter constraints"""
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
            params.get('mu_v', 0.01) > 1e-7,
            params.get('eta_jv', 0.01) > 1e-7
        ]
        return all(constraints)