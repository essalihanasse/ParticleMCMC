import numpy as np


class BootstrapFilter():
    def __init__(self, num_particles):
        """
        Initialize the bootstrap filter.

        Parameters:
        - num_particles: Number of particles in the filter.
        """
        self.num_particles = num_particles

    def run_filter(self, observations, model, S0, V0, r, delta, T, N):
        """
        Run the bootstrap filter on simulated data.

        Parameters:
        - observations: Observed data (e.g., simulated prices).
        - model: Simulation model instance.
        - S0: Initial stock price.
        - V0: Initial variance.
        - r: Risk-free rate.
        - delta: Dividend yield.
        - T: Time horizon.
        - N: Number of time steps.

        Returns:
        - estimated_states: Filtered state estimates.
        """
        dt = T / N
        particles = np.zeros((self.num_particles, len(observations)))
        weights = np.ones(self.num_particles) / self.num_particles

        # Initialize particles
        particles[:, 0] = np.random.normal(V0, 0.1, self.num_particles)

        estimated_states = []

        for t in range(1, len(observations)):
            # Propagate particles
            for i in range(self.num_particles):
                particles[i, t] = particles[i, t-1] + model.kappa * (model.theta - particles[i, t-1]) * dt + \
                                  model.sigma * np.sqrt(max(particles[i, t-1], 0)) * np.random.normal(0, np.sqrt(dt))

                particles[i, t] = max(particles[i, t], 0)

            # Compute weights
            weights *= np.exp(-0.5 * ((observations[t] - particles[:, t]) ** 2))
            weights /= np.sum(weights)

            # Resample particles
            indices = np.random.choice(self.num_particles, self.num_particles, p=weights)
            particles = particles[indices]
            weights.fill(1.0 / self.num_particles)

            # Estimate state
            estimated_states.append(np.mean(particles[:, t]))

        return np.array(estimated_states)
