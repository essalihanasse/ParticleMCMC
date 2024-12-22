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

    def __init__(self, kappa, theta, sigma, rho, eta_s, eta_v, lmda, mu_s, sigma_s, eta_js, sigma_c) -> None:
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

    def __init__(self, kappa, theta, sigma, rho, eta_s, eta_v, lmda, mu_v, eta_jv, sigma_c) -> None:
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
class SimulationSVJC():
    
    def __init__(self, kappa, theta, sigma, rho, eta_s, eta_v, lmda, mu_s, sigma_s, eta_js, sigma_c) -> None:
        """
        Initialize parameters for the SVJC model.
        
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

    def SVJC(self, S0, V0, r, delta, T, N):
        """
        Simulate the SVJC model paths with jumps in returns only.

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
            z_t = np.random.normal(0, 1)
            w_t = np.random.normal(0, 1)

            # Correlated Brownian motion for variance
            w_t = self.rho * z_t + np.sqrt(1 - self.rho**2) * w_t
            
            # Jump frequency (Bernoulli random variable B_t+1)
            B_t1 = np.random.binomial(1, self.lmda * dt)

            # Jump sizes for returns and variance
            J_s = np.random.normal(self.mu_s, self.sigma_s) if B_t1 > 0 else 0
            J_v = np.random.exponential(self.eta_v) if B_t1 > 0 else 0

            # --- Corrected jump compensation term ---
            # Standard Merton jump compensator:  lmda * (exp(mu_s + 0.5*sigma_s^2) - 1)
            lambda_mu_s = self.lmda * (np.exp(self.mu_s + 0.5 * self.sigma_s**2) - 1)

            # Update variance (no jump in variance if you really only want jump in returns, 
            # or incorporate the jump if needed)
            dV = ( self.kappa * (self.theta - V_prev) * dt
                   + self.sigma * np.sqrt(max(V_prev, 0)) * w_t 
                   # + J_v  # uncomment this if you actually want a variance jump
                 )
            V_new = max(V_prev + dV, 0)

            # Update log return using the discretized formula (adding the jump J_s)
            log_return = (
                r - delta - 0.5 * V_prev + self.eta_s * V_prev - lambda_mu_s
            ) * dt + np.sqrt(max(V_prev, 0)) * z_t + J_s * B_t1

            # Update stock price
            S_new = S_prev * np.exp(log_return)

            # Store updated values
            S[t] = S_new
            V[t] = V_new

        return S, V
