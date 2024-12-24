import numpy as np

class PMMH():
    def __init__(self, num_iterations, proposal_std):
        """
        Initialize the PMMH algorithm.

        Parameters:
        - num_iterations: Number of PMMH iterations.
        - proposal_std: Standard deviation for the Gaussian proposal distribution.
        """
        self.num_iterations = num_iterations
        self.proposal_std = proposal_std

    def run(self, observations, model, prior_mean, prior_std, filter_instance, initial_theta):
        """
        Run the PMMH algorithm to estimate theta.

        Parameters:
        - observations: Observed data (e.g., simulated prices).
        - model: Simulation model instance.
        - prior_mean: Mean of the prior distribution for theta.
        - prior_std: Standard deviation of the prior distribution for theta.
        - filter_instance: An instance of the BootstrapFilter.
        - initial_theta: Initial guess for theta.

        Returns:
        - theta_samples: Samples of theta from the posterior distribution.
        """
        current_theta = initial_theta
        current_loglikelihood = self.log_likelihood(observations, model, filter_instance, current_theta)

        theta_samples = [current_theta]

        for _ in range(self.num_iterations):
            proposed_theta = np.random.normal(current_theta, self.proposal_std)

            # Evaluate log likelihood
            proposed_loglikelihood = self.log_likelihood(observations, model, filter_instance, proposed_theta)

            # Compute acceptance probability
            prior_current = -0.5 * ((current_theta - prior_mean) ** 2) / (prior_std ** 2)
            prior_proposed = -0.5 * ((proposed_theta - prior_mean) ** 2) / (prior_std ** 2)

            acceptance_ratio = np.exp(proposed_loglikelihood + prior_proposed - current_loglikelihood - prior_current)

            if np.random.uniform(0, 1) < acceptance_ratio:
                current_theta = proposed_theta
                current_loglikelihood = proposed_loglikelihood

            theta_samples.append(current_theta)

        return np.array(theta_samples)

    def log_likelihood(self, observations, model, filter_instance, theta):
        """
        Compute the log likelihood of the data given the model.

        Parameters:
        - observations: Observed data (e.g., simulated prices).
        - model: Simulation model instance.
        - filter_instance: An instance of the BootstrapFilter.
        - theta: Current value of theta.

        Returns:
        - log_likelihood: The log likelihood of the observations.
        """
        model.theta = theta
        estimated_states = filter_instance.run_filter(observations, model, S0=100, V0=0.04, r=0.03, delta=0.02, T=1, N=len(observations)-1)
        log_likelihood = -0.5 * np.sum((observations - estimated_states) ** 2)
        return log_likelihood
