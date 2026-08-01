```@meta
CurrentModule = KITE
```

# KITE.jl

Quantitative multi-sector Ricardian trade models, solved in changes ("exact hat algebra").

KITE.jl evaluates counterfactual trade policy — tariffs, non-tariff barriers, sanctions, export
subsidies — in the framework of Caliendo & Parro (2015) and the sanctions-coalition extension of
Chowdhry, Hinz, Kamin & Wanner (2022). It is a Julia translation of the R package
[KITE](https://github.com/julianhinz/KITE).

## Installation

```julia
using Pkg
Pkg.develop(path = "path/to/KITE.jl")   # not yet registered
```

## Quick start

```julia
using KITE

b = load_baseline(year = 2022)          # 81 countries × 50 sectors

sc = Scenario(b; label = "25% US tariff on China")
set_tariff!(sc, b, 1.25; from = "CHN", to = "USA", mode = :multiply)

r = update_equilibrium(CaliendoParro2015(), b, sc)
results(r; level = :country)
```

## The model

Households in each country `d` consume a Cobb–Douglas bundle over `J` sectors. Each sector
produces with labour and intermediates from every sector, under Fréchet productivity with shape
`θ_j` — the trade elasticity. Trade costs combine tariffs `τ`, iceberg/non-tariff barriers `κ`
and export taxes or subsidies `ζ`.

Everything is solved in changes, so the technology and geography terms drop out. Writing
`x̂ = x′/x` and `φ̂ = τ̂ κ̂ ζ̂`:

```math
\hat c_d^j = \hat z_d^{j\,-1}\,\hat w_d^{\beta_d^j}
             \Bigl(\prod_k (\hat P_d^k)^{\gamma_d^{kj}}\Bigr)^{1-\beta_d^j}
```
```math
\hat P_d^j = \Bigl(\sum_o \pi_{od}^j \bigl(\hat\varphi_{od}^j\,\hat c_o^j\bigr)^{-\theta_j}\Bigr)^{-1/\theta_j}
```
```math
\pi_{od}^{j\prime} = \pi_{od}^j
    \Bigl(\frac{\hat\varphi_{od}^j\,\hat c_o^j}{\hat P_d^j}\Bigr)^{-\theta_j}
```

with expenditure, income and output closing the system:

```math
X_d^{j\prime} = \alpha_d^j I_d' + \sum_k (1-\beta_d^k)\gamma_d^{jk} Y_d^{k\prime},
\qquad
Y_o^{j\prime} = \sum_d \frac{\pi_{od}^{j\prime}}{\tau_{od}^{j\prime}\zeta_{od}^{j\prime}} X_d^{j\prime}
```

The solver iterates on the wage change `ŵ`. [`CaliendoParro2015`](@ref) updates it from
labour-market clearing; [`ChowdhryHinzKaminWanner2022`](@ref) from the trade-balance
excess-demand function, and additionally computes transfers that equalise the welfare cost
across a sanctioning coalition.

!!! note "Trade elasticity convention"
    `θ` is the standard Fréchet shape parameter: **larger `θ` means more responsive trade**. The
    R implementation inverts the exponents, so its `trade_elasticity` argument behaves as `1/θ`.
    See [Differences from the R implementation](@ref).

## Model-consistent baselines

A [`KiteBaseline`](@ref) satisfies the equilibrium identities at the baseline itself:

```
Y[o,j] = Σ_d π[o,d,j] · X[d,j] / (τ[o,d,j] · ζ[o,d,j])          goods market
X[d,k] = α[d,k] · I[d] + Σ_j input_share[d,k,j] · Y[d,j]        expenditure
I[d]   = VA[d] + R[d] − D[d]                                    income
```

The constructor rejects data that violates them. This is what makes a no-change scenario return
every hat equal to one, in a single iteration:

```julia
r = update_equilibrium(CaliendoParro2015(), b)
r.iterations           # 1
maximum(abs, r.ŵ .- 1) # ~1e-16
```

Raw MRIO data does not satisfy the identities — published final demand, value added and trade
balances are not mutually consistent once run through an input-output loop. [`calibrate`](@ref)
resolves this by making final demand residual (so `α` reproduces observed final demand by
construction) and letting the trade balance close the income identity. Inspect the result with
[`residuals`](@ref).

## Bringing your own data

The long CSV interchange format matches the R package's initial conditions, one file per
variable with its index columns plus `value`:

```julia
b = read_baseline_csv("my_data")   # trade_share.csv, intermediate_share.csv, factor_share.csv,
                                   # expenditure.csv, trade_elasticity.csv, tariff.csv, …
```

Tables may be sparse: absent cells take a documented fill and the number filled is reported.
[`write_baseline`](@ref) writes the same format back.

## Differences from the R implementation

Three defects in the R package change the numbers it produces; all are fixed here. In brief:

1. `cast_variable()` reshapes long tables positionally and silently recycles values when a table
   is sparse — 99.9% of the shipped 2022 trade-share array is wrong.
2. The trade-elasticity exponents are inverted, so it solves a model with elasticity `1/θ`.
3. The baseline is not required to be model-consistent, so a no-change scenario does not
   reproduce it.

Plus several smaller corrections to `ChowdhryHinzKaminWanner2022`. See `NEWS.md` for the full
account with measured magnitudes.

## Validation

The no-change scenario returning unity in one iteration is the headline test. Beyond it: the R
solver, with its inverted exponents patched out, serves as an independent second implementation
and agrees to `1e-15` on wages, prices and trade shares across three scenarios with identical
iteration counts. The suite also checks homogeneity of degree one, market clearing and Walras'
law, numéraire invariance, monotonicity in `θ`, trade diversion, the autarky limit, and
agreement between the iterative and direct inner solvers.

## References

* *Caliendo, L. and Parro, F. (2015).* Estimates of the Trade and Welfare Effects of NAFTA.
  **Review of Economic Studies**, 82(1), 1–44.
* *Chowdhry, S., Hinz, J., Kamin, K. and Wanner, J. (2024).* Brothers in arms: The value of
  coalitions in sanctions regimes. **Economic Policy**, 39(118), 471–512.
* *Fontagné, L., Guimbard, H. and Orefice, G. (2022).* Tariff-based product-level trade
  elasticities. **Journal of International Economics**, 137.
