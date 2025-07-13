import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from typing import Tuple

# Import core functions from the simulation script
# We need to re-define the global variables or pass them as arguments
# to the functions to make them configurable via Gradio.
# For simplicity, I will pass them as arguments to a new wrapper function.

# -----------------------------
# ---- Capital‑market inputs ---
# -----------------------------
# Annualised nominal return expectations
MU_EQUITY_DEFAULT = 0.080   # 8.0 % (Further adjusted for higher equity allocation)
MU_BOND_DEFAULT   = 0.040   # 4.0 % (Further adjusted for higher equity allocation)
SIG_EQUITY_DEFAULT = 0.15   # 15 % st‑dev
SIG_BOND_DEFAULT   = 0.06   # 6 % st‑dev
RHO_DEFAULT        = 0.20   # Equity / bond correlation

# -----------------------------
# ----- Investor profile ------
# -----------------------------
GAMMA_DEFAULT = 4.0            # CRRA utility coefficient
SALARY_0_DEFAULT = 200_000     # Starting wage at simulation year 0
SALARY_GROWTH_DEFAULT = 0.03  # 3 % real growth
CONTRIB_RATE_DEFAULT = 0.20    # 20 % salary deferral
RETIRE_AGE_DEFAULT = 65
CURRENT_AGE_DEFAULT = 45       # So "years to retire" = 20 by default

N_PATHS_DEFAULT = 8_000        # Monte‑Carlo paths
SEED_DEFAULT = 2025

# Helper functions (copied from vanguard_glidepath_sim.py, modified to accept parameters)
def simulate_wealth_paths(w_e: float,
                          years: int,
                          mu_equity: float, mu_bond: float,
                          sig_equity: float, sig_bond: float,
                          rho: float,
                          n_paths: int, seed: int,
                          salary_0: float, salary_growth: float,
                          contrib_rate: float,
                          current_wealth: float
                          ) -> np.ndarray:
    """Simulate wealth accumulation over *years* under constant
    equity weight *w_e*. Contributions are made at START of each year."""
    rng = np.random.default_rng(seed)
    
    COV = np.array([
        [sig_equity**2, rho*sig_equity*sig_bond],
        [rho*sig_equity*sig_bond, sig_bond**2]
    ])

    # draw correlated equity / bond returns
    means = np.array([mu_equity, mu_bond])
    r = rng.multivariate_normal(mean=means, cov=COV, size=(n_paths, years))
    equity_r = r[:, :, 0]
    bond_r   = r[:, :, 1]

    # portfolio return each year
    port_r = w_e * equity_r + (1 - w_e) * bond_r

    # salary path
    t = np.arange(years)
    salary = salary_0 * (1 + salary_growth) ** t  # deterministic wage
    contribs = contrib_rate * salary              # fixed % contributions

    # simulate wealth paths
    W = np.full((n_paths,), current_wealth, dtype=float)
    for yr in range(years):
        W += contribs[yr]           # contribution at beginning of year
        W *= (1 + port_r[:, yr])    # grow for the year
    return W

def expected_utility_terminal_wealth(W: np.ndarray,
                                     gamma: float) -> float:
    if gamma == 1:
        util = np.log(W)
    else:
        # Handle cases where W might be zero or negative, leading to issues with power
        # Replace non-positive values with a small positive number to avoid RuntimeWarning
        W_positive = np.where(W > 0, W, 1e-9) 
        util = (W_positive**(1 - gamma)) / (1 - gamma)
    return np.mean(util)


def find_optimal_equity_weight(years: int,
                               mu_equity: float, mu_bond: float,
                               sig_equity: float, sig_bond: float,
                               rho: float,
                               gamma: float,
                               n_paths: int, seed: int,
                               salary_0: float, salary_growth: float,
                               contrib_rate: float,
                               current_wealth: float,
                               grid: np.ndarray | None = None) -> Tuple[float, float]:
    """Exhaustive search over equity weights to maximise expected utility."""
    if grid is None:
        grid = np.linspace(0.3, 0.95, 14)  # 30 % → 95 % in 5 pp steps
    best_w, best_u = None, -np.inf
    for w in grid:
        W = simulate_wealth_paths(w, years, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                  n_paths, seed, salary_0, salary_growth, contrib_rate, current_wealth)
        u = expected_utility_terminal_wealth(W, gamma)
        if u > best_u:
            best_u, best_w = u, w
    return best_w, best_u


def build_glidepath(max_years: int,
                    mu_equity: float, mu_bond: float,
                    sig_equity: float, sig_bond: float,
                    rho: float,
                    gamma: float,
                    n_paths: int, seed: int,
                    salary_0: float, salary_growth: float,
                    contrib_rate: float,
                    current_wealth: float,
                    grid_start: float, grid_end: float, grid_interval: float) -> pd.DataFrame:
    """Produce Vanguard‑style decreasing equity share from
    *max_years* → 0 to retirement."""
    results = []
    for yrs in range(max_years, -1, -1):
        # Calculate number of steps based on interval
        num_steps = int(np.ceil((grid_end - grid_start) / grid_interval)) + 1
        w_opt, _ = find_optimal_equity_weight(yrs, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                              gamma, n_paths, seed, salary_0, salary_growth, contrib_rate, current_wealth,
                                              grid=np.linspace(grid_start, grid_end, num_steps))
        results.append({'years_to_retire': yrs, 'equity_weight': w_opt})
    return pd.DataFrame(results).set_index('years_to_retire')


def run_simulation(mu_equity: float, mu_bond: float, sig_equity: float, sig_bond: float, rho: float,
                   gamma: float, salary_0: float, salary_growth: float, contrib_rate: float,
                   retire_age: int, current_age: int, n_paths: int, seed: int, current_wealth: float,
                   grid_start: float, grid_end: float, grid_interval: float):

    years_to_retirement = retire_age - current_age
    if years_to_retirement <= 0:
        return "Error: Current age must be less than retirement age.", None, None

    # Calculate optimal equity weight for current years to retirement
    w_star, _ = find_optimal_equity_weight(years_to_retirement, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                           gamma, n_paths, seed, salary_0, salary_growth, contrib_rate, current_wealth)
    optimal_weight_text = f"Optimal equity weight with {years_to_retirement} years to retirement → {w_star:.2%}"

    # Build the full glide path
    gp = build_glidepath(years_to_retirement, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                         gamma, n_paths, seed, salary_0, salary_growth, contrib_rate, current_wealth,
                         grid_start, grid_end, grid_interval)

    # Plot the glide path
    fig, ax = plt.subplots(figsize=(10, 6))
    gp.sort_index().equity_weight.plot(title="Derived Glide Path", ax=ax)
    ax.set_ylabel("Equity share")
    ax.set_xlabel("Years to retirement")
    ax.invert_xaxis()
    ax.grid(True, linestyle='--')  # Add grid lines for better readability
    plt.tight_layout()

    # Simulate wealth paths using the derived glide path (dynamic weights)
    n_paths = n_paths  # Use the user-specified number of paths
    years = years_to_retirement
    wealth_matrix = np.zeros((n_paths, years + 1))
    equity_weights = gp.sort_index(ascending=False)['equity_weight'].values  # years_to_retire from high to low

    # Initial wealth for all paths
    wealth_matrix[:, 0] = current_wealth
    rng = np.random.default_rng(seed)
    COV = np.array([
        [sig_equity**2, rho*sig_equity*sig_bond],
        [rho*sig_equity*sig_bond, sig_bond**2]
    ])
    means = np.array([mu_equity, mu_bond])

    salary = salary_0
    for yr in range(years):
        # Simulate returns for this year
        returns = rng.multivariate_normal(means, COV, size=n_paths)
        equity_r = np.clip(returns[:, 0], -0.4, 0.4)
        bond_r = np.clip(returns[:, 1], -0.2, 0.2)
        w_e = equity_weights[yr]
        # Salary and contribution for this year
        contrib = contrib_rate * salary
        # Update wealth for all paths
        wealth_matrix[:, yr] += contrib
        wealth_matrix[:, yr + 1] = wealth_matrix[:, yr] * (1 + w_e * equity_r + (1 - w_e) * bond_r)
        salary *= (1 + salary_growth)

    # Compute percentiles
    median_wealth = np.median(wealth_matrix[:, 1:], axis=0)
    p10_wealth = np.percentile(wealth_matrix[:, 1:], 10, axis=0)
    p90_wealth = np.percentile(wealth_matrix[:, 1:], 90, axis=0)
    mean_wealth = np.mean(wealth_matrix[:, 1:], axis=0)

    # Plot percentiles
    fig_paths, ax_paths = plt.subplots(figsize=(10, 6))
    x = np.arange(1, years + 1)
    ax_paths.plot(x, median_wealth, label="Median Wealth", color="blue", linewidth=2)
    ax_paths.plot(x, p10_wealth, label="10th Percentile", color="orange", linestyle="--")
    ax_paths.plot(x, p90_wealth, label="90th Percentile", color="green", linestyle="--")
    ax_paths.plot(x, mean_wealth, label="Mean Wealth", color="red", linestyle=":")
    import matplotlib.ticker as mticker
    ax_paths.set_xlabel("Year")
    ax_paths.set_ylabel("Wealth ($)")
    ax_paths.set_title("Simulated Wealth Distribution Using Glide Path (Median, 10th, 90th Percentile, Mean)")
    ax_paths.legend()
    ax_paths.grid(True, linestyle='--')
    ax_paths.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
    plt.tight_layout()

    # Simulate a wealth path using the optimal equity weight
    #W_path = simulate_wealth_paths(w_star, years_to_retirement, mu_equity, mu_bond, sig_equity, sig_bond, rho,
    #                              n_paths, seed, salary_0, salary_growth, contrib_rate)

    # Create a DataFrame to store the year-by-year data
    data = []
    t = np.arange(years_to_retirement)
    salary = salary_0 * (1 + salary_growth) ** t
    contribs = contrib_rate * salary
    W = current_wealth # Initialize W here
    return_list = []
    for yr in range(years_to_retirement):
        r = rng.multivariate_normal(mean=means, cov=COV)
        equity_return = r[0]
        bond_return = r[1]

        # Clip the returns to a reasonable range
        equity_return = np.clip(equity_return, -0.4, 0.4)
        bond_return = np.clip(bond_return, -0.2, 0.2)

        return_list.append(equity_return)
        current_equity_weight = gp.loc[years_to_retirement - yr, 'equity_weight']
        W += contribs[yr]
        W *= (1 + (current_equity_weight * equity_return + (1 - current_equity_weight) * bond_return))
        data.append({
            'Year': yr + 1,
            'Salary': "${:,.2f}".format(salary[yr]),
            'Contribution': "${:,.2f}".format(contribs[yr]),
            'Wealth': "${:,.2f}".format(W),
            'Equity Return': "{:.2%}".format(equity_return),
            'Bond Return': "{:.2%}".format(bond_return),
            'Equity Weight': "{:.2f}".format(current_equity_weight)
        })
    example_path_df = pd.DataFrame(data)

    # Plot the distribution of returns
    fig_returns, ax_returns = plt.subplots(figsize=(10, 6))
    ax_returns.hist(return_list, bins=50)
    ax_returns.set_xlabel("Equity Return")
    ax_returns.set_ylabel("Frequency")
    ax_returns.set_title("Distribution of Equity Returns")
    plt.tight_layout()
    
    return (
        optimal_weight_text,
        fig,
        "The following plot shows an example wealth path using optimal Equity Weights derived from the Glide Path",
        example_path_df,
        "The following plot shows the median, 10th, and 90th percentile of simulated terminal wealth at each year, using the glide path.",
        fig_paths
    )

# Add a detailed introduction using gr.Markdown
introduction_markdown = """
**Vanguard Life-Cycle Investing Model (VLCM) - Didactic Re-implementation**

This application provides an **open-box** re-implementation of the core ideas behind Vanguard's Life-Cycle Investing Model. It aims to illustrate the methodology for determining optimal asset allocation (equity weights) over an investor's lifecycle.

**What it does:**
1.  **Generates Correlated Returns:** Simulates annual equity and bond returns based on user-supplied capital-market assumptions (expected returns, volatilities, and correlation).
2.  **Simulates Wealth Paths:** For a given constant equity weight, it simulates many possible wealth accumulation paths, considering voluntary contributions, salary growth, and time to retirement.
3.  **Computes Expected Utility:** Calculates the expected Constant Relative Risk Aversion (CRRA) utility of terminal wealth for each simulated path. This utility function quantifies investor satisfaction with wealth, accounting for risk aversion.

The CRRA utility function is defined as: $U(W) = \frac{W^{1-\gamma}}{1-\gamma}$ when gamma is not equal to 1.

And for the special case when gamma equals 1: $U(W) = \log(W)$

where:
    W = terminal wealth
        
    gamma = coefficient of relative risk aversion

Example:
    Let's say an investor has a terminal wealth of $100,000 and a risk aversion coefficient (gamma) of 2.
        
    The utility would be calculated as:
        
    U(100000) = (100000^(1-2))/(1-2) = (100000^(-1))/(-1) = -1/100000 = -0.00001
4.  **Searches for Optimal Allocation:** It exhaustively searches over a range of candidate equity weights to find the allocation that maximizes the expected utility for each "years-to-retirement" point.

This is the key step that derives the optimal equity weight for the current years to retirement, which is then used to build a glide path.

The search is structured as follows:
* Inside every inner loop the equity mix stays fixed at w for the entire remaining Y-year horizon.
* Because you repeat that inner loop for every w, the algorithm ends up with one “best” weight w* for that specific starting point.

```
for Y in [40, 39, …, 0]:               # years-to-retirement
    for w in [0%,10%,…,100%]:          # candidate equity weights
        simulate N wealth paths  # 8000 simulation for example
        compute expected utility EY[w]  # average of 8000 utility values based on terminal wealth of each simulated wealth path
    choose w* that maximises EY[w] # this is the optimal equity weight for this Y
    store w* as OPT[Y]
```

More specifically, the algorithm considers each years-to-retirement : counting down years from 40 to 0
* consider different choices of equity weights for the current year, e.g. 10%, 20%, ..., 100%
   * simulate wealth path from this year to the retirement year : e.g. 8000 times
   * in this simplified model the CRRA utility is applied only to the ending (terminal) wealth balance that exists at the moment of retirement, not to the intermediate yearly balances.
   * compute the average terminal wealth balance's utility values of all paths
* find equity weight leading to max average utility values for this year, based on 8000 simulation for each weight

The psedocode above is expanded to the code below:
```
// Outer loop: Iterate through years to retirement, from max_years_to_retirement down to 0
for Y in [max_years_to_retirement, max_years_to_retirement - 1, …, 0]:
    // Initialize best utility found for this Y
    best_expected_utility_for_Y = -infinity
    optimal_equity_weight_for_Y = None

    // Inner loop: Iterate through a predefined grid of candidate equity weights (w)
    // (e.g., 0.30, 0.35, ..., 0.95 as implemented by default)
    for w in [candidate_equity_weights_grid]:
        // Simulate N wealth paths for this fixed w over Y years
        // (N is typically 8000 as per N_PATHS_DEFAULT)
        terminal_wealth_paths = simulate_wealth_paths(w, Y, ...)

        // Compute expected utility EY[w]
        // This is the average of the utility values based on the terminal wealth of each simulated path
        expected_utility_for_w = expected_utility_terminal_wealth(terminal_wealth_paths, gamma)

        // Choose w* that maximizes EY[w] for this Y
        if expected_utility_for_w > best_expected_utility_for_Y:
            best_expected_utility_for_Y = expected_utility_for_w
            optimal_equity_weight_for_Y = w

    // Store the optimal equity weight (w*) for this Y
    store optimal_equity_weight_for_Y as OPT[Y]
```

5.  **Derives Glide Path:** The result is a "glide path" DataFrame, showing the optimal equity allocation as years to retirement decrease. This can be compared with typical glide paths suggested by models like Vanguard's.

**Key Assumptions and Internal Algorithms:**
*   **Capital Market Assumptions:** The model relies on user-defined expected returns, volatilities, and correlation for equities and bonds. These are crucial inputs that drive the simulation.
*   **Monte Carlo Simulation:** It uses Monte Carlo methods to generate a large number of possible future return scenarios, capturing the randomness and correlation of asset returns.

*   **Constant Relative Risk Aversion (CRRA) Utility:** Investor preferences are modeled using a CRRA utility function. This function implies that investors are risk-averse and that their risk aversion decreases as their wealth increases (or remains constant, depending on the specific form). The `gamma` parameter controls the degree of risk aversion.
*   **Wealth Accumulation:** Wealth paths are simulated year by year, starting with the current investment asset value, with contributions made at the beginning of each year and then growing with portfolio returns.
*   **Exhaustive Search:** For each year-to-retirement, the model iterates through a predefined grid of equity weights to find the one yielding the highest expected utility.

**Limitations and Caveats:**
*   **Simplified Model:** This re-implementation is far more compact and simplified compared to Vanguard’s industrial-strength models.
*   **Ignored Factors:** It ignores several real-world complexities such as:
    *   Taxes
    *   Annuitisation options (converting accumulated wealth into a stream of income)
    *   Inflation shocks (only nominal returns are considered, though salary growth is real)
    *   Behavioral biases
    *   Liquidity constraints
*   **Placeholder Assumptions:** The default capital-market assumptions are placeholders and should be replaced with your own research-backed expectations for more realistic results.
*   **No Failure Rate Calculation:** While the original model might compute a 'failure rate', this re-implementation focuses on utility maximization.
*   **No Inflation Adjustment:** All returns and salaries are in nominal terms, without explicit inflation adjustment for the final wealth, though salary growth is real.

This model serves as a didactic tool to understand the principles of life-cycle investing and optimal asset allocation under uncertainty.
"""

with gr.Blocks(css="""
    .gr-form-row p {
        color: #333333 !important; /* Dark gray for better contrast */
    }
    """) as demo:
    with gr.Accordion("Detailed Introduction: Click to expand or collapse ", open=False):
        gr.Markdown(introduction_markdown)
    iface = gr.Interface(
        fn=run_simulation,
        inputs=[
            gr.Slider(minimum=0.01, maximum=0.20, value=MU_EQUITY_DEFAULT, label="Equity Expected Return (annualised nominal)", info="The anticipated average annual return for equities."),
            gr.Slider(minimum=0.01, maximum=0.10, value=MU_BOND_DEFAULT, label="Bond Expected Return (annualised nominal)", info="The anticipated average annual return for bonds."),
            gr.Slider(minimum=0.05, maximum=0.30, value=SIG_EQUITY_DEFAULT, label="Equity Volatility (st-dev)", info="The standard deviation of annual equity returns, representing risk."),
            gr.Slider(minimum=0.01, maximum=0.15, value=SIG_BOND_DEFAULT, label="Bond Volatility (st-dev)", info="The standard deviation of annual bond returns, representing risk."),
            gr.Slider(minimum=-0.5, maximum=0.5, value=RHO_DEFAULT, label="Equity / Bond Correlation", info="The correlation coefficient between equity and bond returns. A higher value means they move more in sync."),
            gr.Slider(minimum=1.0, maximum=10.0, value=GAMMA_DEFAULT, label="CRRA Utility Coefficient (Gamma)", info="Coefficient of Relative Risk Aversion. Higher values indicate greater risk aversion, leading to more conservative allocations."),
            gr.Number(value=SALARY_0_DEFAULT, label="Starting Annual Salary", info="The initial annual salary at the start of the simulation."),
            gr.Slider(minimum=0.00, maximum=0.05, value=SALARY_GROWTH_DEFAULT, label="Annual Salary Growth Rate", info="The annual rate at which the salary is expected to grow."),
            gr.Slider(minimum=0.01, maximum=0.30, value=CONTRIB_RATE_DEFAULT, label="Annual Contribution Rate (% of salary)", info="The percentage of salary contributed to the investment portfolio each year."),
            gr.Slider(minimum=50, maximum=75, value=RETIRE_AGE_DEFAULT, step=1, label="Retirement Age", info="The age at which the investor plans to retire."),
            gr.Slider(minimum=20, maximum=60, value=CURRENT_AGE_DEFAULT, step=1, label="Current Age", info="The investor's current age. Used to calculate years to retirement."),
            gr.Slider(minimum=1000, maximum=20000, value=N_PATHS_DEFAULT, step=1000, label="Number of Monte-Carlo Paths", info="The number of simulation runs to perform for statistical accuracy."),
            gr.Number(value=SEED_DEFAULT, label="Random Seed", info="A seed for the random number generator to ensure reproducible results."),
            gr.Number(value=2000000, label="Current Investment Asset Value", info="The current value of the investor's assets."),
            gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.01, label="Equity Grid Start", info="Starting equity weight for optimal search (e.g., 0.3 for 30%)."),
            gr.Slider(minimum=0.0, maximum=1.0, value=0.95, step=0.01, label="Equity Grid End", info="Ending equity weight for optimal search (e.g., 0.95 for 95%)."),
            gr.Slider(minimum=0.01, maximum=0.20, value=0.05, step=0.01, label="Equity Grid Interval", info="Interval between equity weights (e.g., 0.05 for 5%).")
        ],
        outputs=[
            gr.Textbox(label="Optimal Equity Weight for Current Years to Retirement"),
            gr.Plot(label="Derived Glide Path Plot"),
            gr.Markdown("The following plot shows an example wealth path using optimal Equity Weights derived from the Glide Path"),
            gr.DataFrame(label="Example Wealth Path"),
            gr.Markdown("The following plot shows the median, 10th, and 90th percentile of simulated terminal wealth at each year, using the glide path."),
            gr.Plot(label="Monte Carlo Simulation Percentiles of Wealth Paths"),
        ],
        title="Vanguard Life-Cycle Investing Model (Didactic Re-implementation)",
        description="Adjust the parameters to simulate wealth accumulation and find optimal asset allocations."
    )
    # iface.render() # Removed this line

if __name__ == "__main__":
    demo.launch()
