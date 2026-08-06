# 0.2.0 (unreleased)

## New models

Both follow specifications in the KITE whitepaper that are not in the public R package (the
closed KITE suite implements many more models).

* **`MahlkowWanner2023`** — carbon/energy extension (whitepaper §2.4, eqs. 23–29). Primary
  fossil-fuel sectors extract a natural resource in fixed supply, so their rental price is
  solved for alongside wages and rents enter income. Secondary fossil fuels combine their
  complementary primary fuel with the rest of their input bundle in **Leontief** rather than
  Cobb-Douglas fashion, so fuel cost shares move with relative prices instead of being pinned by
  a fixed exponent — which is what lets fuel use, and hence emissions, respond to trade shocks.
  `fossil_use` reports real secondary-fuel use; `resource_price_change` the rental prices.
  Unlike the whitepaper, tariffs and export subsidies are supported, so the model reduces
  exactly to `CaliendoParro2015` when no fossil sectors are designated.

* **`AntrasChor2018`** — global value chains (whitepaper §4.1, eqs. 34–42). Sourcing becomes
  use-specific: `π[o,d,j,k]` for sector-`j` goods bought by sector `k`, plus `π[o,d,j,C]` for
  final consumption, each with its own price index. Takes a `GVCBaseline`, which re-solves gross
  output against the use-specific goods-market condition and exposes an aggregate-equivalent
  `KiteBaseline` so every results helper keeps working. Tariffs stay product-specific, matching
  how trade policy actually applies. Reduces exactly to `CaliendoParro2015` under
  use-independent sourcing. Note the memory: `π[o,d,j,k]` is 125 MB at 81 countries × 50
  sectors, and the solver holds two such arrays.

## Fixed

* **The convergence criterion missed shocks that move prices but not wages.** The outer loop
  measured only the wage change `ŵ`, following the R implementation. A shock that is symmetric
  across countries leaves `ŵ` at its fixed point from the very first iteration while prices are
  still propagating through the input-output linkages, so the solver stopped immediately and
  reported a price vector that did not satisfy its own input-cost equation. On a four-country
  test a uniform doubling of one sector's trade costs converged in 1 iteration with a **23%**
  violation of the input-cost equation, leaving every other sector's price index at 1. The
  criterion now covers the price block as well; the same case takes 53 iterations and the
  violation falls to 5e-13. This affected `CaliendoParro2015` and `ChowdhryHinzKaminWanner2022`
  as well as the new models, and it affects the R implementation.

# 0.1.0

First release: a Julia translation of the R package
[KITE](https://github.com/julianhinz/KITE) 26.05, implementing `CaliendoParro2015` and
`ChowdhryHinzKaminWanner2022`.

## Corrections to the R implementation

Several issues in the public R package can change the numbers it produces. They are fixed here,
each with a regression test. Results from KITE.jl will **not** always match the R package on the
same inputs; they match a corrected R implementation to machine precision (see
`dev/validate_against_R.R`).

* **The outer convergence criterion monitored only wages.** A shock that is symmetric across
  countries can leave `ŵ` at its fixed point from the first iteration while prices are still
  propagating through the input-output linkages, so the solver stopped early with a price vector
  that did not satisfy its own input-cost equation. The criterion now covers the price block as
  well (see the Fixed section above for measured magnitudes). This affects
  `CaliendoParro2015` and `ChowdhryHinzKaminWanner2022` in the public R package too.

* **The solver's trade-elasticity convention contradicts the shipped data and documentation.**
  R applies `^(-1/θ)` inside the price-index sum and `^(-θ)` outside. That is *not* an algebra
  error: it is algebraically and computationally identical to Caliendo & Parro equations (11)
  and (12) with `θ` replaced by `1/θ`, so the solver's parameter means the **dispersion**
  `1/elasticity`, not the elasticity. It follows Chowdhry et al.'s Appendix B, which is
  self-consistent with that paper's Table A.1 (values 0.04–0.78, i.e. reciprocals of the
  Fontagné et al. elasticities it cites — though Chowdhry et al.'s own main text, eq. 4, uses
  the opposite convention).

  The defect is the mismatch. The shipped `initial_conditions_2022` supplies standard Fontagné
  elasticities, `θ ∈ [1.40, 14.78]`, and `initial_conditions_format.md` documents the field as
  "Fréchet shape parameter (trade elasticity) … typically 1–15" — the opposite of what the
  solver expects. Feeding elasticities into a slot read as `1/elasticity` yields effective
  elasticities of `[0.068, 0.72]`, understating trade reallocation by roughly a factor of `θ²`.
  On a model-consistent economy the measured partial elasticity `dln π / dln τ` is `−0.13`
  against a theoretical `−3.34`.

  Verified directly: the **unmodified** R solver fed `1/θ` reproduces KITE.jl fed `θ` to 1e-11,
  with identical iteration counts. KITE.jl takes `θ` as the standard elasticity, matching the
  KITE whitepaper's equation (13), and a test asserts that raising `θ` strengthens the trade
  response.

  This is not a numerical-stability device. Both forms are CES price indices with
  `d ln P̂ / d ln ĉ_o = π′_od`, so both are perfectly conditioned (sensitivities in `[0,1]`
  summing to one) at every `θ` from 0.2 to 50; the inner power stays representable to `θ ≈ 150`
  under a 10⁶ embargo, and underflow to zero is the correct economic limit anyway. Nor does the
  mis-parameterisation help convergence: on the test fixture the correct `θ` converged in 46
  iterations against 94 for the mis-fed one, and across a correctly parameterised sweep higher
  `θ` converges *faster* (97 iterations at `θ = 0.2` down to 18 at `θ = 30`).

* **The baseline was not required to be model-consistent.** R takes the supplied levels at face
  value. Because the shipped `consumption_share` is built from household consumption only while
  `expenditure` also contains government, investment and inventory demand, `α·I` cannot
  reproduce `X`, and a no-change scenario does not return the baseline. KITE.jl's `calibrate`
  derives final-absorption shares residually, drives residual final demand non-negative, and
  closes the income identity through the trade balance. The `KiteBaseline` constructor refuses
  data that violates the goods-market, expenditure or income identity, so a no-change scenario
  is exact in one iteration.

* **Long tables were reshaped positionally when incomplete.** R's `cast_variable()` builds
  arrays with `array(x[, value], dim = ...)`, where `dim` is the full Cartesian grid implied by
  the index columns. This is harmless when initial conditions are square (complete grids) — as
  in the package's own test fixtures built with `CJ()` — but when a table has fewer rows than
  that grid, R recycles values silently and every entry after the first gap lands in the wrong
  cell. On an incomplete 2022 extract, `trade_share` had 301,871 of 328,050 cells and column
  sums that must equal one ranged from 0.005 to 66; `expenditure` and `consumption_share` were
  each missing 10 cells. KITE.jl places every value by label, fills absent cells with a
  documented default, reports how many were filled, and rejects duplicate or unknown labels.

## Smaller corrections

* `ChowdhryHinzKaminWanner2022` deflates exports and imports by `τ′·ζ′` rather than `τ′` alone,
  matching CP2015 and the paper's own wage equation. Invisible without export taxes or
  subsidies, wrong with them.
* Its transfer step uses the counterfactual trade balance `D′`, not the baseline `D`, which
  disagreed under every `trade_balance_rule` other than `:fixed`.
* Its wage vector is renormalised against the numéraire each iteration, as in CP2015. The R
  version has no normalisation, leaving the wage level pinned only by iteration drift.
* Its inner loop honours `max_inner_iterations`, `inner_tolerance` and `convergence` instead of
  hard-coding 5,000 iterations and an aggregate criterion, and is Jacobi rather than in-place
  Gauss–Seidel, so results no longer depend on the order countries happen to be listed in.
* `convergence = :sample` uses a deterministic stride. R calls `sample()`, so repeated runs of
  the same problem gave different answers.
* Post-processing reads the calibrated baseline directly. R re-runs its expenditure fixed point
  inside `process_results`, which both costs time and lets the reported baseline drift from the
  one the solver used.
* Both models share one array layout. R uses `[country, sector]` in CP2015 and
  `[sector, country]` in CHKW, reconciled by a transpose in post-processing.

## Notes

* Real quantities are invariant to `numeraire` only when the trade balance scales with the price
  level — under `trade_balance_rule = :zero` or `:fixed_global_share`. Under `:fixed` the deficit
  is exogenous in nominal terms and anchors the price level. This is a property of the model,
  not of the solver, and is documented on `SolverSettings`.
* `(destination, sector)` markets with no recorded trade (sector `T`, activities of households,
  in the OECD ICIO) are closed to home supply, `π[d,d,j] = 1`. Expenditure there is zero, so the
  convention only keeps the price index well defined.
* Negative trade shares arising from supply-use balancing items are clipped to zero before
  renormalising; on the 2022 database this touches 8 of 301,871 cells.
