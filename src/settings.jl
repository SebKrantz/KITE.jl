"""
    SolverSettings(; kwargs...)

Numerical settings for [`update_equilibrium`](@ref). Options are validated once, on
construction. The settings are stored in the returned [`KiteResult`](@ref), so a run is fully
reproducible from its own output.

# Keyword Arguments
- `tolerance::Float64 = 1e-6`: outer-loop convergence tolerance, applied to the wage change `ŵ`.
- `inner_tolerance::Float64 = 1e-10`: inner-loop tolerance, applied to gross output `Y′`.
- `max_iterations::Int = 1000`: outer-loop iteration cap.
- `max_inner_iterations::Int = 10_000`: inner-loop iteration cap.
- `vfactor::Float64 = 0.2`: dampening factor for the wage update, in `(0, 1]`. Lower values are
  more stable but slower; raise it for well-behaved scenarios.
- `convergence::Symbol = :root_mean_square`: how the criterion is computed. One of
  `:root_mean_square`, `:aggregate`, `:element_wise`, `:sample`.
- `trade_balance_rule::Symbol = :fixed`: how the trade balance adjusts in the counterfactual.
  One of `:fixed`, `:fixed_country_share`, `:fixed_global_share`, `:zero`.
- `inner_solver::Symbol = :iterative`: `:iterative` runs the warm-started expenditure fixed
  point; `:direct` forms and factorises the `NJ × NJ` linear system instead. The two agree to
  round-off; `:direct` is much more expensive but useful as an independent check.
- `require_inner_convergence::Bool = true`: refuse to declare outer convergence while the inner
  loop is still short of its tolerance.
- `numeraire::Union{Symbol,String} = :world_value_added`: `:world_value_added` holds `Σ VA′`
  fixed; a country code fixes that country's wage. Real quantities are invariant to this choice
  **only when the trade balance scales with the price level** — that is, under
  `trade_balance_rule = :zero` or `:fixed_global_share`. Under `:fixed` the deficit is an
  exogenous *nominal* quantity, so it anchors the price level and the numéraire genuinely
  matters; this is a property of the model, not of the solver.
- `verbose::Int = 1`: `0` silent, `1` a progress line per solve, `2` per-iteration trace.

# Examples
```julia
update_equilibrium(CaliendoParro2015(), b, sc; tolerance = 1e-8, vfactor = 0.3)
update_equilibrium(CaliendoParro2015(), b, sc, SolverSettings(inner_solver = :direct))
```
"""
struct SolverSettings
    tolerance::Float64
    inner_tolerance::Float64
    max_iterations::Int
    max_inner_iterations::Int
    vfactor::Float64
    convergence::Symbol
    trade_balance_rule::Symbol
    inner_solver::Symbol
    require_inner_convergence::Bool
    numeraire::Union{Symbol,String}
    verbose::Int

    function SolverSettings(; tolerance::Real = 1e-6,
                              inner_tolerance::Real = 1e-10,
                              max_iterations::Integer = 1000,
                              max_inner_iterations::Integer = 10_000,
                              vfactor::Real = 0.2,
                              convergence::Symbol = :root_mean_square,
                              trade_balance_rule::Symbol = :fixed,
                              inner_solver::Symbol = :iterative,
                              require_inner_convergence::Bool = true,
                              numeraire::Union{Symbol,AbstractString} = :world_value_added,
                              verbose::Integer = 1)

        convergence in (:root_mean_square, :aggregate, :element_wise, :sample) ||
            error("convergence must be :root_mean_square, :aggregate, :element_wise or " *
                  ":sample; got :$convergence.")
        trade_balance_rule in (:fixed, :fixed_country_share, :fixed_global_share, :zero) ||
            error("trade_balance_rule must be :fixed, :fixed_country_share, " *
                  ":fixed_global_share or :zero; got :$trade_balance_rule.")
        inner_solver in (:iterative, :direct) ||
            error("inner_solver must be :iterative or :direct; got :$inner_solver.")
        if numeraire isa Symbol
            numeraire === :world_value_added ||
                error("numeraire must be :world_value_added or a country code; got :$numeraire.")
        end
        tolerance > 0 || error("tolerance must be positive; got $tolerance.")
        inner_tolerance > 0 || error("inner_tolerance must be positive; got $inner_tolerance.")
        max_iterations > 0 || error("max_iterations must be positive; got $max_iterations.")
        max_inner_iterations > 0 ||
            error("max_inner_iterations must be positive; got $max_inner_iterations.")
        0 < vfactor ≤ 1 || error("vfactor must lie in (0, 1]; got $vfactor.")
        verbose ≥ 0 || error("verbose must be non-negative; got $verbose.")

        num = numeraire isa Symbol ? numeraire : String(numeraire)
        return new(Float64(tolerance), Float64(inner_tolerance), Int(max_iterations),
                   Int(max_inner_iterations), Float64(vfactor), convergence,
                   trade_balance_rule, inner_solver, require_inner_convergence, num,
                   Int(verbose))
    end
end

function Base.show(io::IO, s::SolverSettings)
    print(io, "SolverSettings(tolerance = ", s.tolerance, ", vfactor = ", s.vfactor,
          ", convergence = :", s.convergence, ", trade_balance_rule = :",
          s.trade_balance_rule, ")")
end
