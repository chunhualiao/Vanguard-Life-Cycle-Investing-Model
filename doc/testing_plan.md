# Testing Plan for app.py

## Functions to be tested:

*   `simulate_wealth_paths`
*   `expected_utility_terminal_wealth`
*   `find_optimal_equity_weight`
*   `build_glidepath`
*   `run_simulation`

## Testing Approach:

We will use the `unittest` module to create unit tests for each of the functions listed above. The tests will cover the following aspects:

*   **Input validation:** Ensure that the functions handle invalid inputs gracefully (e.g., negative values, incorrect data types).
*   **Correctness:** Verify that the functions produce the expected outputs for a range of valid inputs.
*   **Edge cases:** Test the functions with edge cases (e.g., zero years to retirement, extreme values for parameters).
*   **Integration:** Test the integration of the functions to ensure they work together correctly.

## Dependencies:

*   `numpy`
*   `pandas`
*   `unittest`

## Test Cases

### `simulate_wealth_paths`
*   Test with different equity weights (`w_e`).
*   Test with different numbers of years.
*   Test with different market parameters (mu, sig, rho).
*   Test with different salary parameters (salary\_0, salary\_growth, contrib\_rate).
*   Test with different numbers of paths and seeds.
*   Test the correlation between equity and bond returns.

### `expected_utility_terminal_wealth`
*   Test with different wealth values (`W`).
*   Test with different risk aversion coefficients (`gamma`).
*   Test with zero or negative wealth values.

### `find_optimal_equity_weight`
*   Test with different market parameters.
*   Test with different risk aversion coefficients.
*   Test with different salary parameters.
*   Test with different numbers of paths and seeds.
*   Test with a custom grid of equity weights.

### `build_glidepath`
*   Test with different maximum years to retirement.
*   Test with different market parameters.
*   Test with different risk aversion coefficients.
*   Test with different salary parameters.
*   Test with different numbers of paths and seeds.

### `run_simulation`
*   Test with different market parameters.
*   Test with different risk aversion coefficients.
*   Test with different salary parameters.
*   Test with different retirement and current ages.
*   Test with different numbers of paths and seeds.
*   Test with current age greater than retirement age.
