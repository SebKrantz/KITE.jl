"""
    KITE

Quantitative multi-sector Ricardian general-equilibrium trade models, solved in changes
("exact hat algebra"). A Julia translation of the R package
[KITE](https://github.com/julianhinz/KITE) (Kiel Institute Trade Policy Evaluation).

Two models are implemented:

* [`CaliendoParro2015`](@ref) — Caliendo & Parro (2015), the multi-sector Ricardian core with
  input-output linkages, tariffs, non-tariff barriers and export taxes/subsidies.
* [`ChowdhryHinzKaminWanner2022`](@ref) — adds sanction-coalition transfers that equalise the
  welfare cost across coalition members.

Workflow:

```julia
using KITE
b  = load_baseline(year = 2022)               # or read_baseline_csv(dir)
sc = Scenario(b)
set_tariff!(sc, b, 1.25; from = "CHN", to = "USA", mode = :multiply)
r  = update_equilibrium(CaliendoParro2015(), b, sc)
results(r; level = :country)
```

A [`KiteBaseline`](@ref) is model-consistent by construction — [`calibrate`](@ref) enforces the
goods-market, expenditure and income identities, so a no-change scenario reproduces the baseline
exactly. See [`update_equilibrium`](@ref) for the solver and its settings.

This port corrects three defects in the R implementation that affect published results; see the
`NEWS.md` file and the "Differences from the R implementation" section of the manual.
"""
module KITE

using LinearAlgebra
using DataFrames
using Printf
using LazyArtifacts
import Artifacts
import CSV
import Tables
import TOML

export KiteModel, CaliendoParro2015, ChowdhryHinzKaminWanner2022, MahlkowWanner2023,
       AntrasChor2018,
       KiteBaseline, GVCBaseline, Scenario, KiteResult, SolverSettings,
       fossil_use, resource_price_change,
       calibrate, residuals, baseline_from_long,
       load_baseline, read_baseline_csv, write_baseline,
       read_baseline_binary, write_baseline_binary,
       set_tariff!, set_ntb!, set_export_subsidy!, set_productivity!, set_population!,
       set_coalition!,
       update_equilibrium, caliendo_parro_2015, chowdhry_hinz_kamin_wanner_2022,
       results, country_results, sector_results, bilateral_results,
       price_index, price_index_change, welfare_change, real_wage_change,
       tariff_revenue, export_subsidy_costs, trade_flows, trade_aggregates

include("settings.jl")   # SolverSettings is a field type of KiteResult, so it comes first
include("types.jl")
include("convergence.jl")
include("baseline.jl")
include("calibrate.jl")
include("scenario.jl")
include("workspace.jl")
include("solve.jl")
include("model_cp2015.jl")
include("model_chkw2022.jl")
include("model_mw2023.jl")
include("model_ac2018.jl")
include("results.jl")
include("output.jl")
include("io.jl")

end # module KITE
