import numpy as np
np.random.seed(0)


class SimulationSV():

    def __init__(self, kappa, theta, sigma, rho, eta_s, eta_v) -> None:
        """
        Initialize parameters for the SVCJ model.

        Parameters:
        - kappa: Speed of mean reversion for variance.
        - theta: Long-term mean of variance.
        - sigma: Volatility of variance (variance of variance).
        - rho: Correlation between Brownian motions dZt and dWt.
        - eta_s: Compensation term for jumps in return.
        - eta_v: Compensation term for jumps in variance.
        - lmda: Intensity of the Poisson process (jumps).
        - mu_s: Mean of the jump in returns.
        - sigma_s: Standard deviation of the jump in returns.
        - correlation_j: Correlation between return and variance jumps.
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.eta_s = eta_s
        self.eta_v = eta_v

    def SV_heston(self, S0, V0, r, delta, T, N):
        """
        Simulate the SV-Heston model paths.

        Parameters:
        - S0: Initial stock price.
        - V0: Initial variance.
        - T: Time horizon.
        - N: Number of time steps.
        """
        # Time points
        dt = T / N
        S = np.zeros(N+1)
        V = np.zeros(N+1)
        S[0] = S0
        V[0] = V0

        # Cholesky decomposition for correlated Brownian motions
        L = np.array([[1, 0], [self.rho, np.sqrt(1 - self.rho**2)]])

        for i in range(N):
            # Generate independent standard normal random variables
            Z = np.random.normal(0, 1, 2)
            dW = np.sqrt(dt) * (L @ Z)  # Correlated Brownian increments

            # Compute the drift term
            mu = (r - delta + self.eta_s * V[i])

            # Update variance (ensure non-negativity)
            V[i+1] = V[i] + self.kappa * (self.theta - V[i]) * dt + self.sigma * np.sqrt(max(V[i], 0)) * dW[1]
            V[i+1] = max(V[i+1], 0)

            # Update stock price
            S[i+1] = S[i] + mu * S[i] * dt + np.sqrt(max(V[i], 0)) * S[i] * dW[0]

        return S, V
        


class SimulationSVJR():

    def __init__(self, kappa, theta, sigma, rho, eta_s, eta_v, lmda,
                 mu_s, sigma_s, eta_js, sigma_c) -> None:
        """
        Initialize parameters for the SVJ model.

        Parameters:
        - kappa: Speed of mean reversion for variance.
        - theta: Long-term mean of variance.
        - sigma: Volatility of variance (variance of variance).
        - rho: Correlation between Brownian motions dZt and dWt.
        - eta_s: Compensation term for jumps in return.
        - eta_v: Compensation term for jumps in variance.
        - lmda: Intensity of the Poisson process (jumps).
        - mu_s: Mean of the jump in returns.
        - sigma_s: Standard deviation of the jump in returns.
        - eta_js: Compensation term for jumps in jump size.
        - sigma_c: Volatility of jump size.
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.eta_s = eta_s
        self.eta_v = eta_v
        self.lmda = lmda
        self.mu_s = mu_s
        self.sigma_s = sigma_s
        self.eta_js = eta_js
        self.sigma_c = sigma_c

    def SVJR(self, S0, V0, r, delta, T, N):
        """
        Simulate the SVJR model paths.

        Parameters:
        - S0: Initial stock price.
        - V0: Initial variance.
        - T: Time horizon.
        - dt: Time step size.
        - r: Risk-free rate.
        - delta: Dividend yield.

        Returns:
        - S: Simulated stock price path.
        - V: Simulated variance path.
        """
        dt = T / N  # Time step size

        # Initialize arrays for stock price and variance
        S = np.zeros(N + 1)
        V = np.zeros(N + 1)

        # Initial values
        S[0] = S0
        V[0] = V0

        # Simulate paths
        for t in range(1, N + 1):
            # Variance and stock price at the previous time step
            V_prev = V[t - 1]
            S_prev = S[t - 1]

            # Generate random variables
            z_t = np.random.normal(0, 1)  # Standard normal for stock
            w_t = np.random.normal(0, 1)  # Standard normal for variance

            # Correlated Brownian motion
            w_t = self.rho * z_t + np.sqrt(1 - self.rho**2) * w_t

            # Jump frequency (Bernoulli random variable B_t+1)
            B_t1 = np.random.binomial(1, self.lmda * dt)

            # Jump sizes for returns
            J_s = np.random.normal(self.mu_s, self.sigma_s) if B_t1 > 0 else 0

            # Compensation for the jump component
            lambda_mu_s = self.lmda * (np.exp(self.mu_s + self.sigma_s**2 / 2) - 1)

            # Update variance using the discretized formula
            dV = self.kappa * (self.theta - V_prev) * dt + self.sigma * np.sqrt(max(V_prev, 0)) * w_t
            V_new = max(V_prev + dV, 0)  # Ensure non-negativity

            # Update log return using the discretized formula
            log_return = (
                r - delta - V_prev / 2 + self.eta_s * V_prev - lambda_mu_s
            ) * dt + np.sqrt(V_prev) * z_t + J_s * B_t1

            # Update stock price
            S_new = S_prev * np.exp(log_return)

            # Store updated values
            S[t] = S_new
            V[t] = V_new

        return S, V


class SimulationSVJV():

    def __init__(self, kappa, theta, sigma, rho, eta_s, eta_v, lmda,
                 mu_v, eta_jv, sigma_c) -> None:
        """
        Initialize parameters for the SVJV model.

        Parameters:
        - kappa: Speed of mean reversion for variance.
        - theta: Long-term mean of variance.
        - sigma: Volatility of variance (variance of variance).
        - rho: Correlation between Brownian motions dZt and dWt.
        - eta_s: Compensation term for jumps in return.
        - eta_v: Compensation term for jumps in variance.
        - lmda: Intensity of the Poisson process (jumps).
        - mu_v: Mean of the jump in variance.
        - eta_jv: Compensation term for jumps in variance.
        - sigma_c: Volatility of jump size.
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.eta_s = eta_s
        self.eta_v = eta_v
        self.lmda = lmda
        self.mu_v = mu_v
        self.eta_jv = eta_jv
        self.sigma_c = sigma_c

    def SVJV(self, S0, V0, r, delta, T, N):
        """
        Simulate the SVJV model paths with jumps in variance only.

        Parameters:
        - S0: Initial stock price.
        - V0: Initial variance.
        - r: Risk-free rate.
        - delta: Dividend yield.
        - T: Time horizon.
        - N: Number of time steps.

        Returns:
        - S: Simulated stock price path.
        - V: Simulated variance path.
        """

        dt = T / N  # Time step size

        # Initialize arrays for stock price and variance
        S = np.zeros(N + 1)
        V = np.zeros(N + 1)

        # Initial values
        S[0] = S0
        V[0] = V0

        # Simulate paths
        for t in range(1, N + 1):
            # Variance and stock price at the previous time step
            V_prev = V[t - 1]
            S_prev = S[t - 1]

            # Generate random variables
            z_t = np.random.normal(0, 1)  # Standard normal for stock
            w_t = np.random.normal(0, 1)  # Standard normal for variance

            # Correlated Brownian motion for variance
            w_t = self.rho * z_t + np.sqrt(1 - self.rho**2) * w_t

            # Jump frequency (Bernoulli random variable B_t+1)
            B_t1 = np.random.binomial(1, self.lmda * dt)  # Probability of a jump in variance

            # Jump sizes for variance
            J_v = np.random.exponential(self.eta_v) if B_t1 > 0 else 0

            # Update variance using the discretized formula
            dV = self.kappa * (self.theta - V_prev) * dt + self.sigma * np.sqrt(max(V_prev, 0)) * w_t + J_v
            V_new = max(V_prev + dV, 0)  # Ensure non-negativity

            # Update log return using the discretized formula (no jumps in returns)
            log_return = (
                r - delta - V_prev / 2 + self.eta_s * V_prev
            ) * dt + np.sqrt(max(V_prev, 0)) * z_t  # No jump term for returns

            # Update stock price
            S_new = S_prev * np.exp(log_return)

            # Store updated values
            S[t] = S_new
            V[t] = V_new

        return S, V


# class SimulationSVCJ:

#     def __init__(self, kappa, theta, sigma, rho, eta_s, eta_v, lmda,
#                  mu_s, sigma_s, eta_js, mu_v, eta_jv, rho_j, sigma_c):
#         """
#         Initialize parameters for the SVCJ model.

#         Parameters:
#         - kappa: Speed of mean reversion for variance.
#         - theta: Long-term mean of variance.
#         - sigma: Volatility of variance (variance of variance).
#         - rho: Correlation between Brownian motions dZt and dWt.
#         - eta_s: Compensation term for jumps in return.
#         - eta_v: Compensation term for jumps in variance.
#         - lmda: Intensity of the Poisson process (jumps).
#         - mu_s: Mean of the jump in returns.
#         - sigma_s: Volatility of jump size in returns.
#         - eta_js: Compensation term for jumps in jump size.
#         - mu_v: Mean of the jump in variance.
#         - eta_jv: Compensation term for jumps in variance.
#         - rho_j: Correlation between return and variance jumps.
#         - sigma_c: Volatility of jump size in variance.
#         """
#         self.kappa = kappa
#         self.theta = theta
#         self.sigma = sigma
#         self.rho = rho
#         self.eta_s = eta_s
#         self.eta_v = eta_v
#         self.lmda = lmda
#         self.mu_s = mu_s
#         self.sigma_s = sigma_s
#         self.eta_js = eta_js
#         self.mu_v = mu_v
#         self.eta_jv = eta_jv
#         self.rho_j = rho_j
#         self.sigma_c = sigma_c

#     def SVCJ(self, S0, V0, r, delta, T, N):
#         """
#         Simulate the SVCJ model paths with jumps in both returns and variance.

#         Parameters:
#         - S0: Initial stock price.
#         - V0: Initial variance.
#         - r: Risk-free rate.
#         - delta: Dividend yield.
#         - T: Time horizon.
#         - N: Number of time steps.

#         Returns:
#         - S: Simulated stock price path.
#         - V: Simulated variance path.
#         """
#         dt = T / N  # Time step size

#         # Initialize arrays for stock price and variance
#         S = np.zeros(N + 1)
#         V = np.zeros(N + 1)

#         # Initial values
#         S[0] = S0
#         V[0] = V0

#         # Simulate paths
#         for t in range(1, N + 1):
#             # Variance and stock price at the previous time step
#             V_prev = V[t - 1]
#             S_prev = S[t - 1]

#             # Generate random variables
#             z_t = np.random.normal(0, 1)  # Standard normal for stock
#             w_t = np.random.normal(0, 1)  # Standard normal for variance

#             # Correlated Brownian motion for variance
#             w_t = self.rho * z_t + np.sqrt(1 - self.rho**2) * w_t

#             # Jump frequency (Bernoulli random variable B_t+1)
#             B_t1 = np.random.binomial(1, self.lmda * dt)  # Probability of a jump

#             # Jump sizes for returns and variance with correlation
#             J_s = (np.random.normal(self.mu_s, self.sigma_s) if B_t1 > 0 else 0) + self.eta_js
#             J_v = (np.random.normal(self.mu_v, self.sigma_c) if B_t1 > 0 else 0) + self.eta_jv

#             # Ensure variance remains non-negative
#             V_prev = max(V_prev, 1e-6)

#             # Update variance using the discretized formula
#             dV = self.kappa * (self.theta - V_prev) * dt + self.sigma * np.sqrt(V_prev) * w_t + J_v
#             V_new = max(V_prev + dV, 0)  # Ensure non-negativity

#             # Cap extreme variance values for numerical stability
#             V_new = min(V_new, 5)

#             # Update log return using the discretized formula
#             log_return = (
#                 (r - delta - V_prev / 2) * dt
#                 + np.sqrt(V_prev) * z_t
#                 + J_s
#             )

#             # Limit extreme changes in stock price to prevent explosion
#             log_return = np.clip(log_return, -0.1, 0.1)

#             # Update stock price
#             S_new = S_prev * np.exp(log_return)

#             # Store updated values
#             S[t] = S_new
#             V[t] = V_new

#         return S, V
#     def compute_option_price(self, S, V, K, tau, r, delta=0):
#         """
#         Compute European call option price using Carr-Madan FFT method.
        
#         Parameters:
#         -----------
#         S : float
#             Current stock price
#         V : float
#             Current variance
#         K : float
#             Strike price
#         tau : float
#             Time to maturity
#         r : float
#             Risk-free rate
#         delta : float, optional
#             Dividend yield
            
#         Returns:
#         --------
#         float
#             Call option price
#         """
#         # FFT parameters
#         N = 4096
#         alpha = 1.5  # Dampening factor
#         eta = 0.25   # Spacing in log-strike domain
#         lambda_ = 2 * np.pi / (N * eta)
#         b = np.pi / eta
        
#         # Grid points
#         v = np.arange(N) * eta
#         k = -b + lambda_ * np.arange(N)
        
#         # Risk-neutral parameters for characteristic function
#         kappa_Q = self.kappa - self.eta_v
#         theta_Q = self.kappa * self.theta / kappa_Q
        
#         # Compute characteristic function
#         u = v - (alpha + 1)*1j
#         d = np.sqrt((self.sigma**2)*(u**2 + u*1j) + 
#                    (kappa_Q - self.rho*self.sigma*u*1j)**2)
#         g = (kappa_Q - self.rho*self.sigma*u*1j - d) / \
#             (kappa_Q - self.rho*self.sigma*u*1j + d)
        
#         C = (kappa_Q * theta_Q / self.sigma**2) * \
#             ((kappa_Q - self.rho*self.sigma*u*1j - d)*tau - 
#              2*np.log((1 - g*np.exp(-d*tau))/(1 - g)))
        
#         D = (kappa_Q - self.rho*self.sigma*u*1j - d) * \
#             (1 - np.exp(-d*tau))/(1 - g*np.exp(-d*tau)) / \
#             self.sigma**2
        
#         cf = np.exp(C + D*V)
        
#         # Modified characteristic function
#         psi = np.exp(-r * tau) * cf / (alpha**2 + alpha - v**2 + 1j*(2*alpha + 1)*v)
        
#         # Apply FFT
#         x = np.exp(1j * b * v) * psi * eta
#         fft_result = np.real(np.fft.fft(x))
        
#         # Get price at desired strike
#         log_strike = np.log(K/S)
#         idx = int((log_strike + b)/lambda_)
#         price = S * np.exp(-delta * tau) * np.exp(-alpha * log_strike) / np.pi * fft_result[idx]
        
#         return np.maximum(price, 0)
import numpy as np
from numba import njit
from scipy.fft import fft

class SimulationSVCJ:
    def __init__(self, kappa, theta, sigma, rho, eta_s, eta_v, lmda, 
                 mu_s, sigma_s, eta_js, mu_v, eta_jv, rho_j, sigma_c):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.eta_s = eta_s
        self.eta_v = eta_v
        self.lmda = lmda
        self.mu_s = mu_s
        self.sigma_s = sigma_s
        self.eta_js = eta_js
        self.mu_v = mu_v
        self.eta_jv = eta_jv
        self.rho_j = rho_j
        self.sigma_c = sigma_c

    def simulate_paths(self, S0, V0, r, delta, T, N):
        """
        Simulate price and variance paths
        """
        dt = 1/252
        steps = int(T/dt)
        
        # Initialize arrays
        S = np.zeros((steps + 1,))
        V = np.zeros((steps + 1,))
        S[0] = S0
        V[0] = V0
        
        # Generate paths
        for t in range(steps):
            # Generate random numbers
            Z_s = np.random.normal(0, 1)
            Z_v = self.rho * Z_s + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1)
            
            # Generate jumps
            if np.random.random() < self.lmda * dt:
                J_s = np.random.normal(self.mu_s, self.sigma_s)
                J_v = np.random.exponential(self.mu_v)
            else:
                J_s = J_v = 0
            
            # Update variance
            V[t+1] = V[t] + self.kappa * (self.theta - V[t]) * dt + \
                     self.sigma * np.sqrt(V[t]) * Z_v * np.sqrt(dt) + J_v
            V[t+1] = max(V[t+1], 1e-7)
            
            # Update price
            S[t+1] = S[t] * np.exp((r - delta - 0.5 * V[t]) * dt + \
                                  np.sqrt(V[t]) * Z_s * np.sqrt(dt) + J_s)
        
        return S, V

    def compute_option_price(self, S, V, K, tau, r):
        """
        Compute European call option price using FFT
        """
        # FFT parameters
        N = 512
        alpha = 1.5
        eta = 0.25
        lambda_ = 2 * np.pi / (N * eta)
        b = np.pi / eta
        
        # Grid points
        v = np.arange(N) * eta
        k = -b + lambda_ * np.arange(N)
        
        # Risk-neutral parameters
        kappa_Q = self.kappa - self.eta_v
        theta_Q = self.kappa * self.theta / kappa_Q
        
        # Characteristic function
        u = v - (alpha + 1)*1j
        d = np.sqrt((self.sigma**2)*(u**2 + u*1j) + 
                   (kappa_Q - self.rho*self.sigma*u*1j)**2)
        
        g = (kappa_Q - self.rho*self.sigma*u*1j - d) / \
            (kappa_Q - self.rho*self.sigma*u*1j + d)
        
        C = (kappa_Q * theta_Q / self.sigma**2) * \
            ((kappa_Q - self.rho*self.sigma*u*1j - d)*tau - 
             2*np.log((1 - g*np.exp(-d*tau))/(1 - g)))
        
        D = (kappa_Q - self.rho*self.sigma*u*1j - d) * \
            (1 - np.exp(-d*tau))/(1 - g*np.exp(-d*tau)) / \
            self.sigma**2
        
        cf = np.exp(C + D*V)
        
        # Modified characteristic function
        psi = np.exp(-r * tau) * cf / (alpha**2 + alpha - v**2 + 1j*(2*alpha + 1)*v)
        
        # FFT
        x = np.exp(1j * b * v) * psi * eta
        fft_result = np.real(fft(x))
        
        # Interpolate for required strike
        log_strike = np.log(K/S)  # Log-moneyness
        idx = int((log_strike + b)/lambda_)
        if 0 <= idx < N:
            price = S * np.exp(-alpha * log_strike) / np.pi * fft_result[idx]
            return max(price, 0)
        return 0.0