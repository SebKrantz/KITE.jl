# KITE.jl

[![CI](https://github.com/SebKrantz/KITE.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/SebKrantz/KITE.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://SebKrantz.github.io/KITE.jl/dev/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Quantitative multi-sector Ricardian trade models, solved in changes.**

KITE.jl evaluates counterfactual trade policy — tariffs, non-tariff barriers, sanctions, export
subsidies — in the general-equilibrium framework of Caliendo & Parro (2015). It is a Julia
translation of the R package [KITE](https://github.com/julianhinz/KITE) (Kiel Institute Trade
Policy Evaluation), rewritten around a model-consistent baseline and dense arrays.

Four models:

| model | adds |
|---|---|
| `CaliendoParro2015` | multi-sector Ricardian core with input-output linkages |
| `ChowdhryHinzKaminWanner2022` | sanction-coalition transfers that equalise the welfare cost |
| `MahlkowWanner2023` | natural-resource rents and a Leontief fossil-fuel nest, for carbon |
| `AntrasChor2018` | use-specific sourcing — global value chains |

The last two were documented in the KITE whitepaper but never implemented in R.

## Installation

```julia
using Pkg
Pkg.develop(path = "path/to/KITE.jl")   # not yet registered
```

## Quick start

```julia
using KITE

b = load_baseline(year = 2022)          # 81 countries × 50 sectors, OECD ICIO

sc = Scenario(b; label = "25% US tariff on China")
set_tariff!(sc, b, 1.25; from = "CHN", to = "USA", mode = :multiply)

r = update_equilibrium(CaliendoParro2015(), b, sc)   # ~0.1 s
results(r; level = :country)            # welfare, prices, trade, income per country
```

Sanctions with burden sharing across a coalition:

```julia
sc = Scenario(b; label = "sanctions on Russia")
set_ntb!(sc, b, 1.5; from = "RUS", to = :all)        # 50% higher trade costs
set_ntb!(sc, b, 1.5; from = :all, to = "RUS")
set_coalition!(sc, b, ["USA", "DEU", "FRA", "GBR"])  # equalises the welfare cost

r = update_equilibrium(ChowdhryHinzKaminWanner2022(), b, sc)
```

Carbon and energy, with fossil-fuel sectors that own a resource and burn fuel in fixed
proportions:

```julia
m = MahlkowWanner2023(b;
        primary = ["B05", "B06"],           # coal, crude oil and gas extraction
        secondary = ["C19" => "B06"],       # refined petroleum ← crude oil
        resource_share = 0.6)               # share of primary value added that is resource rent

r = update_equilibrium(m, b, sc)
fossil_use(r)                                # real fuel use — the emissions driver
resource_price_change(r)
```

Global value chains, where sourcing differs by using sector:

```julia
g = GVCBaseline(b; π_use = π_use, π_fin = π_fin)   # (N,N,J,J) and (N,N,J)
r = update_equilibrium(AntrasChor2018(), g, sc)
r.ext.P̂_use                                  # price of sector-j goods bought by sector k
```

Bring your own data — the long CSV format matches the R package's initial conditions:

```julia
b = read_baseline_csv("my_data")        # trade_share.csv, intermediate_share.csv, …
```

## The baseline is model-consistent by construction

`calibrate` enforces the three equilibrium identities before the solver runs:

```
Y[o,j] = Σ_d π[o,d,j] · X[d,j] / (τ[o,d,j] · ζ[o,d,j])          goods market
X[d,k] = α[d,k] · I[d] + Σ_j input_share[d,k,j] · Y[d,j]        expenditure
I[d]   = VA[d] + R[d] − D[d]                                    income
```

so a no-change scenario reproduces the baseline **exactly, in one iteration** — the invariant
the test suite checks first. Raw MRIO data does not satisfy these identities: final demand is
made residual, and the trade balance closes the income identity.

## Differences from the R implementation

The port fixes three defects that change published results. Each is documented in
[NEWS.md](NEWS.md) and guarded by a regression test.

| | R behaviour | KITE.jl |
|---|---|---|
| **Array casting** | `cast_variable()` reshapes long tables positionally, silently recycling values when a table is sparse. On the 2022 database 26,179 of 328,050 trade-share cells are absent and **99.9% of the resulting array is wrong** (column sums range 0.005–66 instead of 1). | Values are placed by label; absent cells take a documented fill and the count is reported; duplicates and unknown labels are errors. |
| **Trade elasticity** | The solver's parameter is the *dispersion* `1/elasticity` (`^(-1/θ)` inside the price sum, `^(-θ)` outside — CP eq. (11)/(12) reparameterised), following Chowdhry et al.'s Appendix B. But the shipped database and docs supply standard elasticities `θ ∈ [1.4, 14.8]`. The two disagree, so trade responses come out 1–2 orders of magnitude too weak. | `θ` is the standard elasticity, matching the whitepaper's eq. (13). Measured `dln π / dln τ` matches `−θ(1−π)`. |
| **Baseline consistency** | Supplied levels are used as-is, so a no-change scenario does not reproduce the baseline. | `calibrate` enforces the identities; the no-change scenario is exact. |

Smaller corrections, all in the Chowdhry–Hinz–Kamin–Wanner model: exports and imports are
deflated by `τ′ζ′` rather than `τ′` alone; the transfer step uses the counterfactual trade
balance `D′` rather than the baseline `D`; the wage vector is renormalised against the numéraire
each iteration, as in CP2015; and the inner loop honours the solver settings instead of
hard-coding 5,000 iterations and an aggregate criterion. `convergence = :sample` uses a
deterministic stride, so runs are reproducible.

## Validation

* **No-change scenario** returns every hat equal to one, to machine precision, in one iteration
  — on synthetic economies and on the real 81 × 50 baseline.
* **Cross-validation against R.** With its inverted exponents patched out and complete grids
  supplied, the R solver is an independent second implementation. The two agree to **1e-15** on
  wages, prices and trade shares across three scenarios, with identical iteration counts
  (`dev/validate_against_R.R`).
* **Economic properties**: homogeneity of degree one, market clearing and Walras' law at the
  solution, numéraire invariance, monotonicity in `θ`, trade diversion, the autarky limit,
  symmetric-world symmetry, and agreement between the iterative and direct inner solvers.

278 tests in total.

## Performance

The 81 × 50 baseline solves a tariff scenario in about **0.1 s**. Loop-invariant quantities are
hoisted out of the iteration, the per-sector operations are single BLAS `gemv!` calls on
contiguous slices, and a steady-state outer iteration allocates nothing.

## References

* *Caliendo, L. and Parro, F. (2015).* Estimates of the Trade and Welfare Effects of NAFTA.
  **Review of Economic Studies**, 82(1), 1–44.
* *Chowdhry, S., Hinz, J., Kamin, K. and Wanner, J. (2024).* Brothers in arms: The value of
  coalitions in sanctions regimes. **Economic Policy**, 39(118), 471–512.
* *Fontagné, L., Guimbard, H. and Orefice, G. (2022).* Tariff-based product-level trade
  elasticities. **Journal of International Economics**, 137.

## Licence

GPL-3, inherited from the R package this translates. KITE is a product of the Kiel Institute
for the World Economy; see [CITATION.bib](CITATION.bib) and contact `KITE@kielinstitut.de`
regarding commercial use.
