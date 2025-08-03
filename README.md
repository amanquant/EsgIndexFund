# EsgIndexFund
This Python script built in Spyder IDE, implements a custom integer programming solver via Branch and Bound (B&B) with ESG constraints, designed to build an optimal index fund composed of a subset of stocks. 
Here's a breakdown of its purpose and structure:

Model Goal
To select a portfolio of q stocks out of n that:
- Minimizes the average pairwise correlation (diversification objective).
- Satisfies a minimum average return constraint.
- Respects ESG targets.
- Uses binary decision variables to enforce index composition constraints.

Key Components
1. Input Data
prices.txt: historical price data (used to compute correlation matrix).
mom.txt: expected return data for each asset.
esg_scores: hardcoded ESG score for each company.

2. Objective Function
The function to minimize is based on pairwise correlation between selected assets (captured by a matrix rho). The idea is to pick a set of q assets such that their average correlations are minimized (i.e., more diversified).

Model Construction: write_IP_index_fund_with_esg(...)
Builds:
- Objective coefficients c
- Constraint matrix A
- RHS vector b

Constraints include:
- Each position in the index must be filled by one company.
- Each company can appear at most once.
- Only q companies can be selected (sum(y) = q).
- A return constraint: average expected return ≥ target.
(ESG is passed, but not explicitly used in this version—might be integrated or used in future extensions.)

Optimization Engine
Implemented manually via:
- Simplex LP solver: solve_LP(...) with simplex_method(...).
- Branch and Bound: branch_and_bound_with_esg(...), which:
  -- Solves LP relaxations.
  -- Checks integer feasibility.
  -- Branches on fractional variables.
  -- Keeps track of best integer solution and prunes branches.

Output
After solving:
- Selected asset indicators: y_j ∈ {0,1} for each company
- Objective value (minimized average correlation).
- Number of B&B nodes explored.
- Running time.
