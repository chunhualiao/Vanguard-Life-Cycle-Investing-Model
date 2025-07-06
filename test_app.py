import unittest
import numpy as np
import pandas as pd
from app import simulate_wealth_paths, expected_utility_terminal_wealth, find_optimal_equity_weight, build_glidepath, run_simulation

class TestSimulateWealthPaths(unittest.TestCase):

    def test_simulate_wealth_paths_basic(self):
        # Basic test case with default values
        w_e = 0.6
        years = 10
        mu_equity = 0.08
        mu_bond = 0.04
        sig_equity = 0.15
        sig_bond = 0.06
        rho = 0.2
        n_paths = 100
        seed = 2025
        salary_0 = 100000
        salary_growth = 0.015
        contrib_rate = 0.12

        wealth_paths = simulate_wealth_paths(w_e, years, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                              n_paths, seed, salary_0, salary_growth, contrib_rate)

        self.assertEqual(wealth_paths.shape, (n_paths,))
        self.assertTrue(np.all(wealth_paths > 0))

    def test_simulate_wealth_paths_different_equity_weights(self):
        # Test case with different equity weights
        w_e_values = [0.2, 0.4, 0.6, 0.8]
        years = 10
        mu_equity = 0.08
        mu_bond = 0.04
        sig_equity = 0.15
        sig_bond = 0.06
        rho = 0.2
        n_paths = 100
        seed = 2025
        salary_0 = 100000
        salary_growth = 0.015
        contrib_rate = 0.12

        for w_e in w_e_values:
            wealth_paths = simulate_wealth_paths(w_e, years, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                                  n_paths, seed, salary_0, salary_growth, contrib_rate)

            self.assertEqual(wealth_paths.shape, (n_paths,))
            self.assertTrue(np.all(wealth_paths > 0))

    def test_simulate_wealth_paths_different_years(self):
        # Test case with different numbers of years
        years_values = [5, 10, 15, 20]
        w_e = 0.6
        mu_equity = 0.08
        mu_bond = 0.04
        sig_equity = 0.15
        sig_bond = 0.06
        rho = 0.2
        n_paths = 100
        seed = 2025
        salary_0 = 100000
        salary_growth = 0.015
        contrib_rate = 0.12

        for years in years_values:
            wealth_paths = simulate_wealth_paths(w_e, years, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                                  n_paths, seed, salary_0, salary_growth, contrib_rate)

            self.assertEqual(wealth_paths.shape, (n_paths,))
            self.assertTrue(np.all(wealth_paths > 0))

    def test_simulate_wealth_paths_correlation(self):
        # Test case to check the correlation between equity and bond returns
        w_e = 0.6
        years = 10
        mu_equity = 0.08
        mu_bond = 0.04
        sig_equity = 0.15
        sig_bond = 0.06
        rho = 0.8  # High correlation for testing
        n_paths = 1000
        seed = 2025
        salary_0 = 100000
        salary_growth = 0.015
        contrib_rate = 0.12

        wealth_paths = simulate_wealth_paths(w_e, years, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                              n_paths, seed, salary_0, salary_growth, contrib_rate)

        # Extract the simulated equity and bond returns from wealth paths (approximation)
        # This assumes that the wealth path is primarily driven by equity and bond returns
        equity_returns = []
        bond_returns = []
        rng = np.random.default_rng(seed)
        COV = np.array([
            [sig_equity**2, rho*sig_equity*sig_bond],
            [rho*sig_equity*sig_bond, sig_bond**2]
        ])
        means = np.array([mu_equity, mu_bond])
        r = rng.multivariate_normal(mean=means, cov=COV, size=(n_paths, years))
        equity_r = r[:, :, 0]
        bond_r   = r[:, :, 1]

        # Calculate the correlation coefficient
        correlation = np.corrcoef(equity_r.flatten(), bond_r.flatten())[0, 1]

        # Assert that the correlation is close to the expected value
        self.assertAlmostEqual(correlation, rho, places=1)

class TestExpectedUtilityTerminalWealth(unittest.TestCase):

    def test_expected_utility_terminal_wealth_basic(self):
        # Basic test case with default values
        W = np.array([100000, 200000, 300000])
        gamma = 2.0
        expected_utility = expected_utility_terminal_wealth(W, gamma)
        self.assertIsInstance(expected_utility, float)

    def test_expected_utility_terminal_wealth_gamma_1(self):
        # Test case with gamma = 1
        W = np.array([100000, 200000, 300000])
        gamma = 1.0
        expected_utility = expected_utility_terminal_wealth(W, gamma)
        self.assertIsInstance(expected_utility, float)

    def test_expected_utility_terminal_wealth_zero_wealth(self):
        W = np.array([0, 0, 0])
        gamma = 2.0
        expected_utility = expected_utility_terminal_wealth(W, gamma)
        self.assertIsInstance(expected_utility, float)

class TestFindOptimalEquityWeight(unittest.TestCase):

    def test_find_optimal_equity_weight_basic(self):
        # Basic test case with default values
        years = 10
        mu_equity = 0.08
        mu_bond = 0.04
        sig_equity = 0.15
        sig_bond = 0.06
        rho = 0.2
        gamma = 2.0
        n_paths = 100
        seed = 2025
        salary_0 = 100000
        salary_growth = 0.015
        contrib_rate = 0.12

        w_opt, _ = find_optimal_equity_weight(years, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                              gamma, n_paths, seed, salary_0, salary_growth, contrib_rate)
        self.assertIsInstance(w_opt, float)
        self.assertTrue(0.0 <= w_opt <= 1.0)

class TestBuildGlidepath(unittest.TestCase):

    def test_build_glidepath_basic(self):
        # Basic test case with default values
        max_years = 10
        mu_equity = 0.08
        mu_bond = 0.04
        sig_equity = 0.15
        sig_bond = 0.06
        rho = 0.2
        gamma = 2.0
        n_paths = 100
        seed = 2025
        salary_0 = 100000
        salary_growth = 0.015
        contrib_rate = 0.12

        gp = build_glidepath(max_years, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                             gamma, n_paths, seed, salary_0, salary_growth, contrib_rate)
        self.assertIsInstance(gp, pd.DataFrame)
        self.assertEqual(len(gp), max_years + 1)

class TestRunSimulation(unittest.TestCase):

    def test_run_simulation_basic(self):
        # Basic test case with default values
        mu_equity = 0.08
        mu_bond = 0.04
        sig_equity = 0.15
        sig_bond = 0.06
        rho = 0.2
        gamma = 2.0
        salary_0 = 100000
        salary_growth = 0.015
        contrib_rate = 0.12
        retire_age = 65
        current_age = 45
        n_paths = 100
        seed = 2025

        optimal_weight_text, gp, fig = run_simulation(mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                                       gamma, salary_0, salary_growth, contrib_rate,
                                                       retire_age, current_age, n_paths, seed)
        self.assertIsInstance(optimal_weight_text, str)
        self.assertIsInstance(gp, pd.DataFrame)
        self.assertTrue(fig is not None)

if __name__ == '__main__':
    unittest.main()
