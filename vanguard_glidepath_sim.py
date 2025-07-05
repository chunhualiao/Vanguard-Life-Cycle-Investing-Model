
"""vanguard_glidepath_sim.py

A didactic, **open‑box** re‑implementation of the core idea behind
Vanguard’s Life‑Cycle Investing Model (VLCM).

* What it does
  1.  Generates correlated annual equity / bond returns from user‑supplied
     capital‑market assumptions.
  2.  Simulates many wealth paths given a *single* constant equity weight, 
     voluntary contributions, salary growth, and retirement age.
  3.  Computes expected CRRA utility of terminal wealth (or *failure rate*).
  4.  Searches over candidate equity weights to find the utility‑maximising
     allocation for each "years‑to‑retirement" point.
  5.  Spits out a glide‑path DataFrame that you can compare with Vanguard’s.

* Caveats
  - Far more compact than Vanguard’s industrial model.
  - Ignores taxes, annuitisation options, inflation shocks, etc.
  - Capital‑market assumptions here are placeholders—swap in your own.

Python ≥3.9, NumPy, Pandas, and optionally Matplotlib for plotting.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

# -----------------------------
# ---- Capital‑market inputs ---
# -----------------------------
# Annualised nominal return expectations
MU_EQUITY = 0.065   # 6.5 %
MU_BOND   = 0.035   # 3.5 %
SIG_EQUITY = 0.15   # 15 % st‑dev
SIG_BOND   = 0.06   # 6 % st‑dev
RHO        = 0.20   # Equity / bond correlation

COV = np.array([
    [SIG_EQUITY**2, RHO*SIG_EQUITY*SIG_BOND],
    [RHO*SIG_EQUITY*SIG_BOND, SIG_BOND**2]
])

# -----------------------------
# ----- Investor profile ------
# -----------------------------
GAMMA = 4.0            # CRRA utility coefficient
SALARY_0 = 100_000     # Starting wage at simulation year 0
SALARY_GROWTH = 0.015  # 1.5 % real growth
CONTRIB_RATE = 0.12    # 12 % salary deferral
RETIRE_AGE = 65
CURRENT_AGE = 45       # So "years to retire" = 20 by default

N_PATHS = 8_000        # Monte‑Carlo paths
SEED = 2025

rng = np.random.default_rng(SEED)

# -----------------------------
# ---- Helper functions -------
# -----------------------------
def simulate_wealth_paths(w_e: float,
                          years: int,
                          salary_0: float = SALARY_0,
                          salary_growth: float = SALARY_GROWTH
                          ) -> np.ndarray:
    """Simulate wealth accumulation over *years* under constant
    equity weight *w_e*. Contributions are made at START of each year."""
    # draw correlated equity / bond returns
    means = np.array([MU_EQUITY, MU_BOND])
    r = rng.multivariate_normal(mean=means, cov=COV, size=(N_PATHS, years))
    equity_r = r[:, :, 0]
    bond_r   = r[:, :, 1]

    # portfolio return each year
    port_r = w_e * equity_r + (1 - w_e) * bond_r

    # salary path
    t = np.arange(years)
    salary = salary_0 * (1 + salary_growth) ** t  # deterministic wage
    contribs = CONTRIB_RATE * salary              # fixed % contributions

    # simulate wealth paths
    W = np.zeros((N_PATHS,))
    for yr in range(years):
        W += contribs[yr]           # contribution at beginning of year
        W *= (1 + port_r[:, yr])    # grow for the year
    return W

def expected_utility_terminal_wealth(W: np.ndarray,
                                     gamma: float = GAMMA) -> float:
    if gamma == 1:
        util = np.log(W)
    else:
        util = (W**(1 - gamma)) / (1 - gamma)
    return np.mean(util)


def find_optimal_equity_weight(years: int,
                               grid: np.ndarray | None = None) -> Tuple[float, float]:
    """Exhaustive search over equity weights to maximise expected utility."""
    if grid is None:
        grid = np.linspace(0.3, 0.95, 14)  # 30 % → 95 % in 5 pp steps
    best_w, best_u = None, -np.inf
    for w in grid:
        W = simulate_wealth_paths(w, years)
        u = expected_utility_terminal_wealth(W)
        if u > best_u:
            best_u, best_w = u, w
    return best_w, best_u


def build_glidepath(max_years: int = 40) -> pd.DataFrame:
    """Produce Vanguard‑style decreasing equity share from
    *max_years* → 0 to retirement."""
    results = []
    for yrs in range(max_years, -1, -1):
        w_opt, _ = find_optimal_equity_weight(yrs)
        results.append({'years_to_retire': yrs, 'equity_weight': w_opt})
    return pd.DataFrame(results).set_index('years_to_retire')


# -----------------------------
# ---- Quick demo when run ----
# -----------------------------
if __name__ == "__main__":
    demo_years = 15
    w_star, _ = find_optimal_equity_weight(demo_years)
    print(f"Optimal equity weight with {demo_years} years to retirement → {w_star:.2%}")

    gp = build_glidepath(40)
    print("\nFirst few rows of derived glide‑path (yrs→equity_wt):")
    print(gp.head())
    try:
        import matplotlib.pyplot as plt  # optional
        ax = gp.sort_index().equity_weight.plot(title="Derived Glide Path")
        ax.set_ylabel("Equity share")
        ax.set_xlabel("Years to retirement")
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass
