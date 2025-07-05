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
SALARY_0_DEFAULT = 100_000     # Starting wage at simulation year 0
SALARY_GROWTH_DEFAULT = 0.015  # 1.5 % real growth
CONTRIB_RATE_DEFAULT = 0.12    # 12 % salary deferral
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
                          contrib_rate: float
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
    W = np.zeros((n_paths,))
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
                               grid: np.ndarray | None = None) -> Tuple[float, float]:
    """Exhaustive search over equity weights to maximise expected utility."""
    if grid is None:
        grid = np.linspace(0.3, 0.95, 14)  # 30 % → 95 % in 5 pp steps
    best_w, best_u = None, -np.inf
    for w in grid:
        W = simulate_wealth_paths(w, years, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                  n_paths, seed, salary_0, salary_growth, contrib_rate)
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
                    contrib_rate: float) -> pd.DataFrame:
    """Produce Vanguard‑style decreasing equity share from
    *max_years* → 0 to retirement."""
    results = []
    for yrs in range(max_years, -1, -1):
        w_opt, _ = find_optimal_equity_weight(yrs, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                              gamma, n_paths, seed, salary_0, salary_growth, contrib_rate)
        results.append({'years_to_retire': yrs, 'equity_weight': w_opt})
    return pd.DataFrame(results).set_index('years_to_retire')


def run_simulation(mu_equity: float, mu_bond: float, sig_equity: float, sig_bond: float, rho: float,
                   gamma: float, salary_0: float, salary_growth: float, contrib_rate: float,
                   retire_age: int, current_age: int, n_paths: int, seed: int):

    years_to_retirement = retire_age - current_age
    if years_to_retirement <= 0:
        return "Error: Current age must be less than retirement age.", None, None

    # Calculate optimal equity weight for current years to retirement
    w_star, _ = find_optimal_equity_weight(years_to_retirement, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                                           gamma, n_paths, seed, salary_0, salary_growth, contrib_rate)
    optimal_weight_text = f"Optimal equity weight with {years_to_retirement} years to retirement → {w_star:.2%}"

    # Build the full glide path
    gp = build_glidepath(years_to_retirement, mu_equity, mu_bond, sig_equity, sig_bond, rho,
                         gamma, n_paths, seed, salary_0, salary_growth, contrib_rate)

    # Plot the glide path
    fig, ax = plt.subplots(figsize=(10, 6))
    gp.sort_index().equity_weight.plot(title="Derived Glide Path", ax=ax)
    ax.set_ylabel("Equity share")
    ax.set_xlabel("Years to retirement")
    ax.invert_xaxis()
    plt.tight_layout()

    return optimal_weight_text, gp, fig

# Gradio Interface
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
        gr.Number(value=SEED_DEFAULT, label="Random Seed", info="A seed for the random number generator to ensure reproducible results.")
    ],
    outputs=[
        gr.Textbox(label="Optimal Equity Weight for Current Years to Retirement"),
        gr.DataFrame(label="Derived Glide Path"),
        gr.Plot(label="Derived Glide Path Plot")
    ],
    title="Vanguard Life-Cycle Investing Model (Didactic Re-implementation)",
    description="Adjust the parameters to simulate wealth accumulation and find optimal asset allocations."
)

# Add a detailed introduction using gr.Markdown
introduction_markdown = """
**Vanguard Life-Cycle Investing Model (VLCM) - Didactic Re-implementation**

This application provides an **open-box** re-implementation of the core ideas behind Vanguard's Life-Cycle Investing Model. It aims to illustrate the methodology for determining optimal asset allocation (equity weights) over an investor's lifecycle.

**What it does:**
1.  **Generates Correlated Returns:** Simulates annual equity and bond returns based on user-supplied capital-market assumptions (expected returns, volatilities, and correlation).
2.  **Simulates Wealth Paths:** For a given constant equity weight, it simulates many possible wealth accumulation paths, considering voluntary contributions, salary growth, and time to retirement.
3.  **Computes Expected Utility:** Calculates the expected Constant Relative Risk Aversion (CRRA) utility of terminal wealth for each simulated path. This utility function quantifies investor satisfaction with wealth, accounting for risk aversion.
4.  **Searches for Optimal Allocation:** It exhaustively searches over a range of candidate equity weights to find the allocation that maximizes the expected utility for each "years-to-retirement" point.
5.  **Derives Glide Path:** The result is a "glide path" DataFrame, showing the optimal equity allocation as years to retirement decrease. This can be compared with typical glide paths suggested by models like Vanguard's.

**Key Assumptions and Internal Algorithms:**
*   **Capital Market Assumptions:** The model relies on user-defined expected returns, volatilities, and correlation for equities and bonds. These are crucial inputs that drive the simulation.
*   **Monte Carlo Simulation:** It uses Monte Carlo methods to generate a large number of possible future return scenarios, capturing the randomness and correlation of asset returns.
*   **Constant Relative Risk Aversion (CRRA) Utility:** Investor preferences are modeled using a CRRA utility function. This function implies that investors are risk-averse and that their risk aversion decreases as their wealth increases (or remains constant, depending on the specific form). The `gamma` parameter controls the degree of risk aversion.
*   **Wealth Accumulation:** Wealth paths are simulated year by year, with contributions made at the beginning of each year and then growing with portfolio returns.
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

with gr.Blocks() as demo:
    gr.Markdown(introduction_markdown)
    gr.CSS("""
    .gr-form-row .gr-info-text {
        color: #333333 !important; /* Dark gray for better contrast */
    }
    """)
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
            gr.Number(value=SEED_DEFAULT, label="Random Seed", info="A seed for the random number generator to ensure reproducible results.")
        ],
        outputs=[
            gr.Textbox(label="Optimal Equity Weight for Current Years to Retirement"),
            gr.DataFrame(label="Derived Glide Path"),
            gr.Plot(label="Derived Glide Path Plot")
        ],
        title="Vanguard Life-Cycle Investing Model (Didactic Re-implementation)",
        description="Adjust the parameters to simulate wealth accumulation and find optimal asset allocations."
    )
    # iface.render() # Removed this line

if __name__ == "__main__":
    demo.launch()
