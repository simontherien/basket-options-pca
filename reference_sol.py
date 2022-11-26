import numpy as np
from tqdm.autonotebook import tqdm

class ReferenceSol:
    def __init__(
        self,
        T: float,
        S0: float,
        corr_matrix: np.ndarray,
        vol: np.ndarray,
        rfr: float,
        K: float,
        nsim: int
    ) -> None:
        """
        Reference solution class constructor

        Parameters
        ----------
        T : float
            Expiry of option in days
        S0 : float
            Initial asset price
        corr_matrix : np.ndarray
            Correlation matrix
        vol : np.ndarray
            Volatility of assets
        rfr : float
            Risk free rate of return
        K : float
            Strike price of basket option
        nsim : int
            Number of simulations required for computing expectation of option payoff
        """
        self.T = int(T)
        self.S0 = S0
        self.corr_matrix = corr_matrix
        self.nassets = corr_matrix.shape[0]
        self.vol = vol
        self.rfr = rfr
        self.K = K
        self.nsim = nsim

    def compute_asset_paths(self) -> np.ndarray:
        """
        Computes correlated asset paths

        Returns
        -------
        np.ndarray
            Array of prices across the time horizon
        """
        R = np.linalg.cholesky(self.corr_matrix)  # Decompose correlation matrix
        prices = np.full((self.nassets, self.T), self.S0)

        # TODO
        #assert np.array_equal(self.corr_matrix,np.dot(R, R.T.conj()))  # Validate cholesky decomposition

        for t in range(1, self.T):
            xi = np.random.standard_normal(self.nassets)  # Create N(0,1) noise vector
            sigma = np.inner(xi, R)  # Obtain correlated noise
            # Sample path for each asset
            for n in range(self.nassets):
                dt = 1/self.T
                S = prices[n, t-1]
                v = self.vol[n]
                s = sigma[n]

                # Generate t+1 asset price with risk neutral geometric brownian motion
                prices[n, t] = S * np.exp((self.rfr - 0.5 * v**2) * dt + v * np.sqrt(dt) * s)
        
        return prices
    
    def compute_basket_payoff(self) -> float:
        """
        Run simulations and compute discounted basket option payoff assuming equal weights

        Returns
        -------
        float
            Option price at time 0
        """
        sim_prices_list = []
        # Run simulations
        for i in tqdm(range(self.nsim), desc="Running simulations"):
            sim_prices_list.append(self.compute_asset_paths())
        
        # Keep only prices at expiry
        sim_prices = np.array(sim_prices_list)
        terminal_prices = sim_prices[:,:,-1]
        # Compute payoff
        payoff = np.maximum(terminal_prices - self.K, 0)
        mean_payoff = np.mean(payoff, axis=0)
        # Discount to time 0
        disc_factor = np.exp(-self.rfr * self.T)

        return np.sum(mean_payoff) * disc_factor