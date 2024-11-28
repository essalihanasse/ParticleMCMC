import numpy as np

class Simulation():

    def __init__(self, kappa, teta, sigma, rho, eta_s, eta_v) -> None:
        self.kappa = kappa
        self.teta = teta
        self.sigma = sigma
        self.rho = rho
        self.eta_s = eta_s
        self.eta_v = eta_v


    def SV_heston(self, S0, V0, T, N, M, dt):
        # Time points
        time_points = np.linspace(0, T, N+1)
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

            # Update variance (ensure non-negativity)
            V[i+1] = V[i] + self.kappa * (self.theta - V[i]) * dt + self.sigma * np.sqrt(max(V[i], 0)) * dW[1]
            V[i+1] = max(V[i+1], 0)

            # Update stock price
            S[i+1] = S[i] + self.mu * S[i] * dt + np.sqrt(max(V[i], 0)) * S[i] * dW[0]

        return time_points, S, V
        
    # def SV_heston(self, S0, v0, T, N, M):
    #     """
    #     Simulate the data using the SV Heston model
    #     """
    #     dt = T / N
    #     S = np.zeros((M, N+1))
    #     v = np.zeros((M, N+1))
    #     S[:, 0] = S0
    #     v[:, 0] = v0
        
    #     Z1 = np.random.normal(0, 1, (M, N))
    #     Z2 = np.random.normal(0, 1, (M, N))
    #     Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2  # Correlated noise
        
    #     for t in range(N):
    #         v[:, t+1] = np.maximum(v[:, t] + self.kappa * (self.theta - v[:, t]) * dt + 
    #                             self.sigma * np.sqrt(v[:, t] * dt) * Z2[:, t], 0)
    #         S[:, t+1] = S[:, t] * np.exp((self.mu - 0.5 * v[:, t]) * dt + 
    #                                     np.sqrt(v[:, t] * dt) * Z1[:, t])
        
    #     return S, v