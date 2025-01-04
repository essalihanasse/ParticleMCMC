import numpy as np
# class ModelDiagnostics:
#     def compute_pricing_errors(self, true_prices, model_prices, by_moneyness=True):
#         errors = true_prices - model_prices
#         rmse = np.sqrt(np.mean(errors**2))
        
#         if by_moneyness:
#             moneyness_bins = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
#             rmse_by_moneyness = []
#             for i in range(len(moneyness_bins)-1):
#                 mask = (moneyness >= moneyness_bins[i]) & \
#                        (moneyness < moneyness_bins[i+1])
#                 rmse_by_moneyness.append(
#                     np.sqrt(np.mean(errors[mask]**2)))
#             return rmse, rmse_by_moneyness
#         return rmse

#     def analyze_convergence(self, chains):
#         # Gelman-Rubin diagnostic
#         W = np.mean(np.var(chains, axis=1))
#         B = np.var(np.mean(chains, axis=1))
#         var_theta = (1 - 1/len(chains[0]))*W + B
#         R_hat = np.sqrt(var_theta/W)
#         return R_hat

import numpy as np

class ModelDiagnostics:
    def compute_pricing_errors(self, true_prices, model_prices, by_moneyness=True):
        errors = true_prices - model_prices
        rmse = np.sqrt(np.mean(errors**2))
        return rmse

    def analyze_convergence(self, chains):
        # Compute Gelman-Rubin statistic
        n = chains.shape[1]  # number of iterations
        m = chains.shape[0]  # number of chains
        
        # Compute between-chain variance
        chain_means = np.mean(chains, axis=1)
        overall_mean = np.mean(chain_means, axis=0)
        B = n * np.var(chain_means, axis=0)
        
        # Compute within-chain variance
        W = np.mean(np.var(chains, axis=1), axis=0)
        
        # Compute variance estimate
        var_plus = ((n-1)/n * W + B/n)
        
        # Compute R-hat
        R_hat = np.sqrt(var_plus / W)
        
        return np.mean(R_hat)