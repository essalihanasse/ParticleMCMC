import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from scipy.stats import norm
import warnings

class OptimizedPMMH:
    """
    Particle Marginal Metropolis-Hastings implementation with orthogonal sampling
    and adaptive proposals for improved mixing and convergence.
    """
    
    def __init__(self, num_iterations=500, num_chains=5, num_vertical=10,
                 num_horizontal=1, use_orthogonal=True, burnin=100,
                 target_acceptance=0.234):
        """
        Initialize PMMH sampler with improved defaults.
        
        Args:
            num_iterations (int): Number of MCMC iterations
            num_chains (int): Number of parallel chains
            num_vertical (int): Number of vertical moves before horizontal swap
            num_horizontal (int): Number of horizontal swap attempts
            use_orthogonal (bool): Whether to use orthogonal sampling
            burnin (int): Number of burn-in iterations
            target_acceptance (float): Target acceptance rate for adaptation
        """
        self.num_iterations = num_iterations
        self.num_chains = num_chains
        self.num_vertical = num_vertical
        self.num_horizontal = num_horizontal
        self.use_orthogonal = use_orthogonal
        self.burnin = burnin
        self.target_acceptance = target_acceptance
        self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        # Initialize adaptation parameters
        self.adaptation_start = int(0.1 * num_iterations)
        self.adaptation_window = 50
        
    def run(self, observations, particle_filter, initial_params, progress_callback=None):
        """
        Main run method that chooses between orthogonal and adaptive MCMC.
        
        Args:
            observations (dict): Observation data including prices and options
            particle_filter (ParticleFilter): Particle filter instance
            initial_params (dict): Initial parameter values
            progress_callback (callable): Optional callback for progress updates
        
        Returns:
            tuple: (chains array, acceptance rates array)
        """
        if self.use_orthogonal:
            return self._run_orthogonal(observations, particle_filter, 
                                      initial_params, progress_callback)
        else:
            return self._run_adaptive(observations, particle_filter, 
                                    initial_params, progress_callback)

    def _run_orthogonal(self, observations, particle_filter, initial_params, 
                       progress_callback=None):
        """
        Run orthogonal MCMC sampling with improved parallel chain handling.
        """
        n_params = len(initial_params)
        chains = np.zeros((self.num_chains, self.num_iterations, n_params))
        param_names = list(initial_params.keys())
        
        # Initialize multiple chains with slight perturbations
        current_params = []
        current_lls = []
        for i in range(self.num_chains):
            perturbed_params = self._perturb_initial_params(initial_params, scale=0.1)
            current_params.append(perturbed_params)
            ll = particle_filter.compute_likelihood(observations, perturbed_params)
            current_lls.append(ll)
        
        accepted = np.zeros(self.num_chains)
        proposal_scales = self._initialize_proposal_scales()
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for iter in range(self.num_iterations):
                # Vertical moves (within-chain updates)
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
                                proposal_scales,
                                iter
                            )
                        )
                    
                    # Collect results and update chains
                    for chain, future in enumerate(futures):
                        new_params, new_ll, was_accepted = future.result()
                        current_params[chain] = new_params
                        current_lls[chain] = new_ll
                        
                        # Store parameters
                        for j, param_name in enumerate(param_names):
                            chains[chain, iter, j] = new_params[param_name] # Apply scaling factor here
                        
                        accepted[chain] += was_accepted
                        
                        if progress_callback:
                            progress_callback(1)
                
                # Horizontal moves (between-chain swaps)
                if iter % (self.num_vertical * self.num_horizontal) == 0:
                    self._perform_horizontal_moves(current_params, current_lls)
                
                # Adapt proposal scales after burn-in
                if iter > self.burnin and iter % self.adaptation_window == 0:
                    self._adapt_proposal_scales(proposal_scales, accepted, iter)
        
        acceptance_rates = accepted / self.num_iterations
        return chains, acceptance_rates

    def _run_adaptive(self, observations, particle_filter, initial_params, 
                     progress_callback=None):
        """
        Run adaptive MCMC sampling with improved covariance estimation.
        """
        n_params = len(initial_params)
        chains = np.zeros((self.num_chains, self.num_iterations, n_params))
        param_names = list(initial_params.keys())
        
        # Initialize covariance matrix
        proposal_cov = self._initialize_proposal_covariance(n_params)
        adaptation_count = 0
        
        current_params = initial_params.copy()
        current_ll = particle_filter.compute_likelihood(observations, current_params)
        
        accepted = 0
        
        for iter in range(self.num_iterations):
            # Update proposal covariance after burn-in
            if iter > self.burnin and iter % self.adaptation_window == 0:
                proposal_cov = self._update_proposal_covariance(
                    chains[0, max(0, iter-500):iter],
                    adaptation_count
                )
                adaptation_count += 1
            
            # Propose new parameters
            proposed_params = self._propose_parameters_adaptive(
                current_params, 
                proposal_cov,
                iter
            )
            
            if not self._check_constraints(proposed_params):
                continue
            
            try:
                proposed_ll = particle_filter.compute_likelihood(
                    observations, 
                    proposed_params
                )
                
                # Accept/reject step
                log_alpha = float(proposed_ll - current_ll)
                if np.log(np.random.random()) < log_alpha:
                    current_params = proposed_params
                    current_ll = proposed_ll
                    accepted += 1
                
                # Store current state
                for j, param_name in enumerate(param_names):
                    chains[0, iter, j] = current_params[param_name] * 10   # Apply scaling factor here
                
                if progress_callback:
                    progress_callback(1)
                    
            except Exception as e:
                warnings.warn(f"Error in likelihood computation: {str(e)}")
                continue
        
        acceptance_rate = accepted / self.num_iterations
        return chains, np.array([acceptance_rate])

    def _initialize_proposal_scales(self):
        """Initialize proposal scales with improved defaults."""
        return {
            'kappa': 0.02,
            'theta': 0.0005,
            'sigma': 0.005,
            'rho': 0.01,
            'eta_s': 0.02,
            'eta_v': 0.01,
            'lmda': 0.01,
            'mu_s': 0.001,
            'sigma_s': 0.002,
            'eta_js': 0.001,
            'mu_v': 0.001,
            'eta_jv': 0.001,
            'rho_j': 0.01,
            'sigma_c': 0.02,
            'V0': 0.0005,
            'r': 0.0002,
            'delta': 0.0002
        }

    def _perturb_initial_params(self, params, scale=0.1):
        """Perturb initial parameters for multiple chains."""
        perturbed = params.copy()
        for param in perturbed:
            if param in ['rho', 'rho_j']:
                continue  # Skip correlation parameters
            perturbed[param] *= (1 + scale * np.random.randn())
        return perturbed

    def _perform_horizontal_moves(self, current_params, current_lls):
        """Perform horizontal (between-chain) moves."""
        for i in range(self.num_chains - 1):
            log_ratio = current_lls[i+1] - current_lls[i]
            if np.log(np.random.random()) < log_ratio:
                current_params[i], current_params[i+1] = \
                    current_params[i+1], current_params[i]
                current_lls[i], current_lls[i+1] = \
                    current_lls[i+1], current_lls[i]

    def _adapt_proposal_scales(self, scales, accepted, iter):
        """Adapt proposal scales based on acceptance rates."""
        if iter <= self.burnin:
            return
        
        # Calculate acceptance rate for each chain
        window_acceptance = accepted / iter  # This is now an array of acceptance rates
        
        # Take mean acceptance rate across all chains
        mean_acceptance = np.mean(window_acceptance)
        
        for param in scales:
            if mean_acceptance < self.target_acceptance:
                scales[param] *= 0.9
            else:
                scales[param] *= 1.1

    def _initialize_proposal_covariance(self, n_params):
        """Initialize proposal covariance matrix."""
        return np.eye(n_params) * 0.0001

    def _update_proposal_covariance(self, chain_history, adaptation_count):
        """Update proposal covariance using chain history."""
        cov = np.cov(chain_history.T)
        scale = 2.4 ** 2 / chain_history.shape[1]
        return scale * cov + 1e-6 * np.eye(cov.shape[0])

    def _propose_parameters(self, current_params, proposal_scales, iter=0):
        """Propose new parameters with improved transformations and adaptive scaling."""
        proposed = {}
        
        # Adjust scales based on iteration to improve early exploration
        scale_factor = 0.1
        if iter < self.burnin:
            scale_factor = max(0.1, min(1.0, iter / self.burnin))
        
        for param, value in current_params.items():
            scale = proposal_scales[param] * scale_factor
            
            try:
                if param in ['kappa', 'theta', 'sigma', 'eta_s', 'eta_v', 'lmda',
                           'sigma_s', 'sigma_c', 'mu_v', 'eta_jv', 'V0']:
                    # Log-normal proposals with bounds for positive parameters
                    log_value = np.log(max(value, 1e-7))
                    proposed_log = log_value + np.random.normal(0, min(scale, 0.5))
                    proposed[param] = np.exp(proposed_log)
                    
                elif param in ['rho', 'rho_j']:
                    # Improved correlation parameter proposals
                    if abs(value) > 0.99:  # Handle boundary cases
                        value = np.sign(value) * 0.99
                    z = np.arctanh(value)
                    proposed_z = z + np.random.normal(0, min(scale, 0.2))
                    proposed[param] = np.tanh(proposed_z)
                    
                else:
                    # Normal proposals with reasonable bounds
                    proposed[param] = value + np.random.normal(0, scale)
                    
                    # Add specific bounds for certain parameters
                    if param in ['r', 'delta']:
                        proposed[param] = max(0, min(0.2, proposed[param]))
                    
            except Exception as e:
                # Fallback to current value if proposal fails
                proposed[param] = value
                warnings.warn(f"Parameter proposal failed for {param}: {str(e)}")
        
        return proposed

    def _propose_parameters_adaptive(self, current_params, proposal_cov, iter):
        """Propose parameters using adaptive multivariate normal."""
        param_names = list(current_params.keys())
        current_values = np.array([current_params[p] for p in param_names])
        
        # Generate proposal in transformed space
        transformed = self._transform_parameters(current_values)
        proposed_transformed = transformed + np.random.multivariate_normal(
            np.zeros(len(transformed)), 
            proposal_cov
        )
        
        # Transform back to original space
        proposed_values = self._inverse_transform_parameters(proposed_transformed)
        
        return dict(zip(param_names, proposed_values))

    def _transform_parameters(self, params):
        """Transform parameters to unconstrained space."""
        transformed = params.copy()
        for i, param in enumerate(params):
            if param > 0:  # Positive parameters
                transformed[i] = np.log(param)
            elif -1 < param < 1:  # Correlation parameters
                transformed[i] = np.arctanh(param)
        return transformed

    def _inverse_transform_parameters(self, transformed_params):
        """Transform parameters back to constrained space."""
        params = transformed_params.copy()
        for i, param in enumerate(transformed_params):
            if param > 0:  # Positive parameters
                params[i] = np.exp(param)
            elif -1 < param < 1:  # Correlation parameters
                params[i] = np.tanh(param)
        return params

    def _check_constraints(self, params):
        """Check parameter constraints with improved conditions."""
        try:
            constraints = [
                params['kappa'] > 0,  # Strict positivity
                params['theta'] > 0,
                params['sigma'] > 0,
                abs(params['rho']) < 1,  # Correlation bounds
                params['lmda'] >= 0,  # Non-negative jump intensity
                params['sigma_s'] > 0,
                params['sigma_c'] > 0,
                abs(params.get('rho_j', 0)) < 1,
                params.get('V0', 1e-4) > 0,
                params.get('r', 0) >= 0,
                params.get('delta', 0) >= 0,
                params.get('mu_v', 0.01) > 0,
                params.get('eta_jv', 0.01) > 0,
                
                # Feller condition
                # 2 * params['kappa'] * params['theta'] > params['sigma']**2
            ]
            return all(constraints)
            
        except KeyError:
            return False
        except Exception as e:
            warnings.warn(f"Error checking constraints: {str(e)}")
            return False

    def _vertical_move(self, observations, particle_filter, current_params,
                      current_ll, proposal_scales, iter):
        """Execute single vertical MCMC move with improved error handling."""
        try:
            # Propose parameters
            proposed_params = self._propose_parameters(
                current_params,
                proposal_scales,
                iter
            )
            
            if not self._check_constraints(proposed_params):
                print("Constraints not met")
                return current_params, current_ll, False
            
            # Compute likelihood
            proposed_ll = particle_filter.compute_likelihood(
                observations,
                proposed_params
            )
            
            # Accept/reject
            log_alpha = float(proposed_ll - current_ll)
            
            if np.isfinite(log_alpha) and np.log(np.random.random()) < log_alpha:
                return proposed_params, proposed_ll, True
            
            return current_params, current_ll, False
            
        except Exception as e:
            warnings.warn(f"Error in vertical move: {str(e)}")
            return current_params, current_ll, False
