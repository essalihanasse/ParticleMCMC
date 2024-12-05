# ParticleMCMC

## Outline for project:
  - Re-create the data for the project by  use the model parameter estimates of the paper and use the same moneyness-maturity grid when simulating the option prices. We will use a smaller dataset as it is time consuming to estimate the model over a large period of time and with 30 options per day.
  - Implement a bootstrap filter for the model under consideration (by setting the parameters to reasonable values)
  - Implement a PMMH-type algorithm to estimate the theta parameter; the article talks about an ‘adaptive’ version, where you gradually learn the covariance matrix of the proposal law (a Gaussian random walk); note that this is already implemented in particles, if you want to use it
  - Compare the performance of this PMMH with the approach proposed in this article (orthogonal MCMC); you can also add SMC^2 to the comparison (again if you have time).
