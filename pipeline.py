import numpy as np
from tqdm import tqdm

class EstimationPipeline:
    def __init__(self, particle_filter, pmmh):
        self.particle_filter = particle_filter
        self.pmmh = pmmh

    def run(self, data, initial_params):
        """Run the full estimation pipeline"""
        print("\nInitializing PMMH estimation...")
        print(f"Number of particles: {self.particle_filter.num_particles}")
        print(f"Number of PMMH iterations: {self.pmmh.num_iterations}")
        print(f"Number of parallel chains: {self.pmmh.num_chains}")
        
        # Store full chain history and diagnostics
        with tqdm(total=self.pmmh.num_iterations * self.pmmh.num_chains,
                 desc="PMMH Progress") as pbar:
            chains, acceptance_rates = self.pmmh.run(
                data,
                self.particle_filter,
                initial_params,
                progress_callback=lambda x: pbar.update(1)
            )

        # Convert to numpy array and compute statistics
        chains = np.array(chains)
        burn_in = int(0.5 * chains.shape[1])  # 50% burn-in
        
        # Calculate posterior means
        posterior_means = {}
        param_names = list(initial_params.keys())
        chain_means = np.mean(chains[:, burn_in:, :], axis=(0, 1))
        
        for name, value in zip(param_names, chain_means):
            posterior_means[name] = value

        # Calculate convergence diagnostics
        r_hat = self._compute_gelman_rubin(chains)
        print(f"\nMean acceptance rate: {np.mean(acceptance_rates):.2%}")
        
        print("\nGelman-Rubin diagnostics (R̂):")
        for name, r in zip(param_names, r_hat):
            print(f"{name}: {r:.3f}")

        # Store particle filter state history if available
        state_history = getattr(self.particle_filter, 'state_history', None)

        return {
            'posterior_means': posterior_means,
            'chains': chains,
            'acceptance_rates': acceptance_rates,
            'r_hat': dict(zip(param_names, r_hat)),
            'burn_in': burn_in,
            'state_history': state_history,
            'convergence_metrics': {
                'mean_acceptance': np.mean(acceptance_rates),
                'r_hat_max': np.max(r_hat)
            }
        }

    def _compute_gelman_rubin(self, chains):
        """Compute Gelman-Rubin convergence diagnostic"""
        M, N, D = chains.shape  # M chains, N iterations, D parameters
        
        # Compute between-chain variance
        chain_means = np.mean(chains, axis=1)  # Shape: (M, D)
        overall_means = np.mean(chain_means, axis=0)  # Shape: (D,)
        B = N * np.var(chain_means, axis=0, ddof=1)  # Shape: (D,)
        
        # Compute within-chain variance
        W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)  # Shape: (D,)
        
        # Compute variance estimate
        var_est = (N - 1) * W / N + B / N
        
        # Compute R-hat
        r_hat = np.sqrt(var_est / W)
        
        return r_hat