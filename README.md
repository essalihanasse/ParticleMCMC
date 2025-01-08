# 🚀 Fast Filtering with Large Option Panels: Implications for Asset Pricing

Welcome to the repository for **"Fast Filtering with Large Option Panels: Implications for Asset Pricing"**. This project introduces a computationally efficient particle MCMC framework to analyze large option datasets, tackling challenges in dynamic option pricing and asset valuation.

## ✨ Key Features

- 📊 **Dynamic Models**: Implementation of models like Heston's square root and Duffie-Pan-Singleton double-jump models.
- 🔧 **Particle Filtering**: Optimized filtering using particle weights based on quantiles.
- 📈 **Empirical Analysis**: Results derived from extensive datasets (S&P500 returns and options).
- 🎛️ **Configurable Pipeline**: Command-line interface for customizing estimation parameters and time horizons.

## 🛠️ Built With

- **NumPy** 🧮
- **SciPy** 🔬
- **Matplotlib** 📊
- **argparse** 🎯

## 🚀 Getting Started

### Prerequisites

Ensure you have:

- Python 3.9+
- Required libraries (install using the command below)

```bash
pip install -r requirements.txt
```

### Command Line Usage

The estimation pipeline can be customized through command line arguments. Here are some common usage patterns:

1. Basic usage with default settings:
```bash
python main.py
```

2. Customize particle filter and MCMC parameters:
```bash
python main.py --num-particles 200 --mcmc-iterations 150 --num-chains 8
```

3. Adjust time horizon and granularity:
```bash
python main.py --time-horizon 0.15873 --time-steps 300
```

### Available Command Line Arguments

- `--num-particles`: Number of particles for filtering (default: 100)
- `--mcmc-iterations`: Number of MCMC iterations (default: 100)
- `--num-chains`: Number of parallel MCMC chains (default: 5)
- `--burnin`: Number of burn-in iterations (default: 10)
- `--vertical-moves`: Number of vertical moves in PMMH (default: 10)
- `--time-horizon`: Time horizon T in years (default: 20/252 ≈ 0.079365)
- `--time-steps`: Number of time steps N (default: 252)

## 🧑‍💻 Project Outline

The project follows these key steps:

1. **Data Recreation**: We simulate option prices using model parameter estimates from the paper, following the same moneyness-maturity grid. For computational efficiency, we use a reduced dataset size while maintaining the essential characteristics of the original study.

2. **Bootstrap Filter Implementation**: A bootstrap filter is implemented for the model with carefully chosen parameter values. This forms the foundation for our particle filtering approach.

3. **PMMH Algorithm**: We implement a Particle Marginal Metropolis-Hastings (PMMH) algorithm for parameter estimation, featuring:
   - Adaptive covariance matrix learning for the proposal distribution
   - Integration with the particles library for efficient sampling
   - Comparison with orthogonal MCMC and SMC² approaches

### Model Configuration

The model parameters can be fine-tuned to match different market conditions:

- Volatility parameters (κ, θ, σ, ρ)
- Jump parameters (λ, μ_s, σ_s, μ_v)
- Risk premiums (η_s, η_v, η_js, η_jv)

## 🤝 Collaborators

- @adevilde (Alice Devilder)
- @S-Amorotti (Sean Amorotti)
- @essalihanasse (Anasse Essalih)

## 🔬 Advanced Usage

For more sophisticated analysis, you can chain multiple parameters:

```bash
python main.py --num-particles 200 --mcmc-iterations 150 --num-chains 8 \
               --burnin 20 --vertical-moves 15 --time-horizon 0.15873 \
               --time-steps 300
```

This flexibility allows researchers to experiment with different configurations while maintaining the robust foundation of the original study.