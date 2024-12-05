# 🚀 Fast Filtering with Large Option Panels: Implications for Asset Pricing

Welcome to the repository for **"Fast Filtering with Large Option Panels: Implications for Asset Pricing"**. This project introduces a computationally efficient particle MCMC framework to analyze large option datasets, tackling challenges in dynamic option pricing and asset valuation.

## ✨ Key Features

- 📊 **Dynamic Models**: Implementation of models like Heston's square root and Duffie-Pan-Singleton double-jump models.
- 🔧 **Particle Filtering**: Optimized filtering using particle weights based on quantiles.
- 📈 **Empirical Analysis**: Results derived from extensive datasets (S&P500 returns and options).

## 🛠️ Built With

- **NumPy** 🧮
- **SciPy** 🔬
- **Matplotlib** 📊
- **Particles** 🎲

## 🚀 Getting Started

### Prerequisites

Ensure you have:

- Python 3.9+
- Required libraries (install using the command below).

```bash
pip install -r requirements.txt
```

## 🧑‍💻 Outline for project:
  - Re-create the data for the project by  use the model parameter estimates of the paper and use the same moneyness-maturity grid when simulating the option prices. We will use a smaller dataset as it is time consuming to estimate the model over a large period of time and with 30 options per day.
  - Implement a bootstrap filter for the model under consideration (by setting the parameters to reasonable values)
  - Implement a PMMH-type algorithm to estimate the theta parameter; the article talks about an ‘adaptive’ version, where you gradually learn the covariance matrix of the proposal law (a Gaussian random walk); note that this is already implemented in particles, if you want to use it
  - Compare the performance of this PMMH with the approach proposed in this article (orthogonal MCMC); you can also add SMC^2 to the comparison (again if you have time).

### 🤝 Collaborators
@adevilde (Alice Devilder)
@S-Amorotti (Sean Amorotti)