# Core types.
#
# Field and local-variable names follow the notation of Caliendo & Parro (2015) and the KITE
# whitepaper: π trade shares, γ intermediate shares, α consumption shares, β value-added
# shares, θ trade elasticities, τ tariffs, ζ export subsidies/taxes. Counterfactual levels
# carry a prime (X′), changes carry a hat (ŵ, ĉ, P̂). Binding a local named `π` shadows
# `Base.π`; KITE never needs the circle constant (the Fréchet gamma factor cancels in changes).
# All exported names, keyword arguments and DataFrame columns are ASCII snake_case.
#
# ── Array layout ──────────────────────────────────────────────────────────────────────────
# Julia is column-major, so `A[o, d, j]` lives at `o + N*(d-1) + N^2*(j-1)`. Storing the
# bilateral tensors as (N, N, J) makes `@view A[:, :, j]` a *contiguous* N×N matrix, so the
# per-sector price-index and output updates are single `gemv!` calls with no copy, and each
# slice (52 KB at N = 81) is L2-resident.
#
# The IO tensors are stored country-first as (N, J, J) indexed [d, k, j] with k the input
# sector and j the output sector. Both hot contractions then run over contiguous N-vectors:
#
#   input cost    Σ_k γ[d, k, j] · log P̂[d, k]        — fixed j, sum over k
#   intermediates Σ_j input_share[d, k, j] · Y[d, j]  — fixed k, sum over j
#
# so `γ` and `input_share` share one layout and no permuted duplicate is needed.

"""
    KiteModel

Abstract supertype for KITE model variants. Concrete subtypes are singletons that select the
solver's wage-update and transfer behaviour by multiple dispatch:
[`CaliendoParro2015`](@ref) and [`ChowdhryHinzKaminWanner2022`](@ref).
"""
abstract type KiteModel end

"""
    CaliendoParro2015()

Multi-sector Ricardian model with input-output linkages of Caliendo & Parro (2015). The wage
iterate is updated from labour-market clearing, `ŵ_o ← (Σ_j β_o^j Y_o^{j′}) / (w_o L_o)`,
log-dampened by `vfactor` and normalised so world value added is the numéraire.

Supports tariffs, non-tariff barriers, export taxes/subsidies, and exogenous productivity and
labour-supply shocks.
"""
struct CaliendoParro2015 <: KiteModel end

"""
    ChowdhryHinzKaminWanner2022()

Extends [`CaliendoParro2015`](@ref) with the sanction-coalition transfer scheme of Chowdhry,
Hinz, Kamin & Wanner (2022/2024). Transfers `T` among coalition members `S` satisfy

    Σ_{d ∈ S} T_d = 0        and        Î_d / P̂_d  equal for all d ∈ S,

so every member bears the same proportional welfare cost. The wage iterate is updated from the
trade-balance excess-demand function rather than from labour-market clearing.

Set the coalition with [`set_coalition!`](@ref). With an empty coalition the model solves the
same equilibrium as [`CaliendoParro2015`](@ref).
"""
struct ChowdhryHinzKaminWanner2022 <: KiteModel end

_model_name(::CaliendoParro2015) = "caliendo_parro_2015"
_model_name(::ChowdhryHinzKaminWanner2022) = "chowdhry_hinz_kamin_wanner_2022"

"""
    KiteBaseline

A calibrated, **model-consistent** initial equilibrium: `N` countries, `J` sectors, the
behavioural shares, the policy levels, and a set of mutually consistent nominal levels.

Consistency means the three equilibrium identities hold at the baseline itself,

    Y[o, j] = Σ_d π[o, d, j] · X[d, j] / (τ[o, d, j] · ζ[o, d, j])       (goods market)
    X[d, k] = α[d, k] · I[d] + Σ_j input_share[d, k, j] · Y[d, j]        (expenditure)
    I[d]    = VA[d] + R[d] − D[d],   VA[d] = Σ_j β[d, j] · Y[d, j]       (income)

and the inner constructor rejects data that violates them. This invariant is what makes a
no-change scenario reproduce the baseline exactly, in a single iteration. Use
[`calibrate`](@ref) to build one from raw data and [`residuals`](@ref) to inspect the three
identity errors.

# Fields
- `N::Int`, `J::Int`: number of countries and sectors.
- `countries::Vector{String}`, `sectors::Vector{String}`: labels, in array order.
- `π::Array{Float64,3}`: `(N, N, J)` `[o, d, j]` trade shares; `Σ_o π[:, d, j] = 1`.
- `γ::Array{Float64,3}`: `(N, J, J)` `[d, k, j]` intermediate input shares; `Σ_k γ[d, :, j] = 1`
  (or `0` for inactive sectors), `k` the input and `j` the output sector.
- `input_share::Array{Float64,3}`: `(N, J, J)` `[d, k, j] = (1 − β[d, j]) · γ[d, k, j]`.
- `α::Matrix{Float64}`: `(N, J)` final-absorption shares; `Σ_j α[d, :] = 1`.
- `β::Matrix{Float64}`: `(N, J)` value-added share of gross output, in `(0, 1)`.
- `θ::Vector{Float64}`: `(J)` trade elasticities (Fréchet shape), `> 0`.
- `τ::Array{Float64,3}`: `(N, N, J)` tariff multiplier `1 + rate`, `≥ 1`.
- `ζ::Array{Float64,3}`: `(N, N, J)` export tax (`> 1`) or subsidy (`< 1`) multiplier.
- `X::Matrix{Float64}`, `Y::Matrix{Float64}`: `(N, J)` expenditure and gross output.
- `I::Vector{Float64}`: `(N)` total final absorption (income).
- `R::Vector{Float64}`: `(N)` net tariff revenue plus export tax/subsidy receipts.
- `VA::Vector{Float64}`: `(N)` value added, `= Σ_j β[d, j] · Y[d, j]`.
- `D::Vector{Float64}`: `(N)` trade **surplus**, subtracted from income; `Σ_d D ≈ 0`.

Non-tariff barriers appear only as the change `κ̂` in a [`Scenario`](@ref), never in levels, so
there is no `κ` field.
"""
struct KiteBaseline
    N::Int
    J::Int
    countries::Vector{String}
    sectors::Vector{String}
    country_index::Dict{String,Int}
    sector_index::Dict{String,Int}
    π::Array{Float64,3}
    γ::Array{Float64,3}
    input_share::Array{Float64,3}
    α::Matrix{Float64}
    β::Matrix{Float64}
    θ::Vector{Float64}
    τ::Array{Float64,3}
    ζ::Array{Float64,3}
    X::Matrix{Float64}
    Y::Matrix{Float64}
    I::Vector{Float64}
    R::Vector{Float64}
    VA::Vector{Float64}
    D::Vector{Float64}

    function KiteBaseline(countries, sectors, π, γ, α, β, θ, τ, ζ, X, Y, I, R, VA, D;
                          atol::Float64 = 1e-8, check::Bool = true)
        N = length(countries)
        J = length(sectors)
        countries = String.(countries)
        sectors = String.(sectors)

        allunique(countries) || error("countries must be unique.")
        allunique(sectors) || error("sectors must be unique.")
        _check_size(π, (N, N, J), "π")
        _check_size(γ, (N, J, J), "γ")
        _check_size(τ, (N, N, J), "τ")
        _check_size(ζ, (N, N, J), "ζ")
        _check_size(α, (N, J), "α")
        _check_size(β, (N, J), "β")
        _check_size(X, (N, J), "X")
        _check_size(Y, (N, J), "Y")
        _check_size(θ, (J,), "θ")
        for (v, nm) in ((I, "I"), (R, "R"), (VA, "VA"), (D, "D"))
            _check_size(v, (N,), nm)
        end

        all(>(0), θ) || error("θ must be strictly positive; got a minimum of $(minimum(θ)).")
        all(x -> 0 < x < 1, β) || error("β must lie strictly in (0, 1); got extrema $(extrema(β)).")
        all(≥(0), π) || error("π must be non-negative; got a minimum of $(minimum(π)).")
        all(>(0), τ) || error("τ must be strictly positive; got a minimum of $(minimum(τ)).")
        all(>(0), ζ) || error("ζ must be strictly positive; got a minimum of $(minimum(ζ)).")

        input_share = similar(γ)
        @inbounds for j in 1:J, k in 1:J
            @views @. input_share[:, k, j] = (1 - β[:, j]) * γ[:, k, j]
        end

        b = new(N, J, countries, sectors,
                Dict(c => i for (i, c) in enumerate(countries)),
                Dict(s => i for (i, s) in enumerate(sectors)),
                Array{Float64,3}(π), Array{Float64,3}(γ), input_share,
                Matrix{Float64}(α), Matrix{Float64}(β), Vector{Float64}(θ),
                Array{Float64,3}(τ), Array{Float64,3}(ζ),
                Matrix{Float64}(X), Matrix{Float64}(Y),
                Vector{Float64}(I), Vector{Float64}(R),
                Vector{Float64}(VA), Vector{Float64}(D))

        if check
            res = residuals(b)
            m = max(res.goods_market, res.expenditure, res.income)
            m ≤ atol || error(
                "KiteBaseline is not model-consistent: goods-market residual " *
                "$(res.goods_market), expenditure residual $(res.expenditure), income " *
                "residual $(res.income) (tolerance $atol). Build the baseline with " *
                "`calibrate` to enforce the equilibrium identities.")
        end
        return b
    end
end

function _check_size(x, dims, name)
    size(x) == dims && return nothing
    throw(DimensionMismatch("$name is $(size(x)), expected $dims."))
end

function Base.show(io::IO, b::KiteBaseline)
    print(io, "KiteBaseline(", b.N, " countries × ", b.J, " sectors)")
end

function Base.show(io::IO, ::MIME"text/plain", b::KiteBaseline)
    res = residuals(b)
    println(io, "KiteBaseline: ", b.N, " countries × ", b.J, " sectors")
    println(io, "  countries : ", _preview(b.countries))
    println(io, "  sectors   : ", _preview(b.sectors))
    @printf(io, "  value added   %.4g (total)\n", sum(b.VA))
    @printf(io, "  expenditure   %.4g (total)\n", sum(b.X))
    @printf(io, "  trade balance %.4g (sum, should be ≈ 0)\n", sum(b.D))
    @printf(io, "  θ ∈ [%.3g, %.3g]\n", minimum(b.θ), maximum(b.θ))
    @printf(io, "  residuals: goods market %.2e, expenditure %.2e, income %.2e",
            res.goods_market, res.expenditure, res.income)
end

function _preview(v::Vector{String}, n::Int = 4)
    length(v) ≤ n && return join(v, ", ")
    return join(v[1:n], ", ") * ", … (" * string(length(v)) * " total)"
end

"""
    Scenario(b::KiteBaseline; label = "scenario")

A counterfactual policy experiment, held as dense arrays of the same shape as the baseline and
initialised to no change. Mutate it with [`set_tariff!`](@ref), [`set_ntb!`](@ref),
[`set_export_subsidy!`](@ref), [`set_productivity!`](@ref), [`set_population!`](@ref) and
[`set_coalition!`](@ref).

# Fields
- `τ′::Array{Float64,3}`: `(N, N, J)` counterfactual tariff multiplier (starts at `b.τ`).
- `ζ′::Array{Float64,3}`: `(N, N, J)` counterfactual export tax/subsidy (starts at `b.ζ`).
- `κ̂::Array{Float64,3}`: `(N, N, J)` non-tariff-barrier *change* (starts at `1`).
- `ẑ::Matrix{Float64}`: `(N, J)` productivity change (starts at `1`); divides input cost.
- `L̂::Vector{Float64}`: `(N)` labour-supply change (starts at `1`).
- `coalition::Vector{Bool}`: `(N)` coalition membership, used by
  [`ChowdhryHinzKaminWanner2022`](@ref) only (starts all `false`).
- `label::String`: a description carried through to results.

Fields are dense rather than `nothing`-able so the solver has no branches in its hot loop;
uniformly-unchanged arrays are detected once and skipped wholesale.
"""
mutable struct Scenario
    τ′::Array{Float64,3}
    ζ′::Array{Float64,3}
    κ̂::Array{Float64,3}
    ẑ::Matrix{Float64}
    L̂::Vector{Float64}
    coalition::Vector{Bool}
    label::String
end

function Scenario(b::KiteBaseline; label::AbstractString = "scenario")
    return Scenario(copy(b.τ), copy(b.ζ), ones(b.N, b.N, b.J), ones(b.N, b.J),
                    ones(b.N), falses(b.N), String(label))
end

function Base.show(io::IO, sc::Scenario)
    print(io, "Scenario(\"", sc.label, "\")")
end

function Base.show(io::IO, ::MIME"text/plain", sc::Scenario)
    println(io, "Scenario: \"", sc.label, "\"")
    @printf(io, "  tariff τ′        ∈ [%.4g, %.4g]\n", extrema(sc.τ′)...)
    @printf(io, "  export subsidy ζ′∈ [%.4g, %.4g]\n", extrema(sc.ζ′)...)
    println(io, "  ntb change κ̂     : ", count(!=(1.0), sc.κ̂), " of ", length(sc.κ̂), " cells changed")
    println(io, "  productivity ẑ   : ", count(!=(1.0), sc.ẑ), " of ", length(sc.ẑ), " cells changed")
    println(io, "  population L̂     : ", count(!=(1.0), sc.L̂), " of ", length(sc.L̂), " countries changed")
    print(io, "  coalition        : ", count(sc.coalition), " members")
end

"""
    KiteResult{M<:KiteModel}

The solved counterfactual equilibrium returned by [`update_equilibrium`](@ref). Changes carry a
hat, counterfactual levels a prime.

# Fields
- `model::M`, `baseline::KiteBaseline`, `scenario::Scenario`, `settings::SolverSettings`
- `ŵ::Vector{Float64}`: `(N)` wage change.
- `ĉ::Matrix{Float64}`, `P̂::Matrix{Float64}`: `(N, J)` input-cost and price-index change.
- `π′::Array{Float64,3}`: `(N, N, J)` counterfactual trade shares.
- `X′::Matrix{Float64}`, `Y′::Matrix{Float64}`: `(N, J)` expenditure and gross output.
- `I′`, `VA′`, `D′`, `T′::Vector{Float64}`: `(N)` income, value added, trade balance, transfers
  (`T′` is all zeros for [`CaliendoParro2015`](@ref)).
- `converged::Bool`, `criterion::Float64`, `iterations::Int`, `inner_iterations::Int`,
  `elapsed::Float64`

Use [`results`](@ref) to turn this into tidy `DataFrame`s.
"""
struct KiteResult{M<:KiteModel}
    model::M
    baseline::KiteBaseline
    scenario::Scenario
    settings::SolverSettings
    ŵ::Vector{Float64}
    ĉ::Matrix{Float64}
    P̂::Matrix{Float64}
    π′::Array{Float64,3}
    X′::Matrix{Float64}
    Y′::Matrix{Float64}
    I′::Vector{Float64}
    VA′::Vector{Float64}
    D′::Vector{Float64}
    T′::Vector{Float64}
    converged::Bool
    criterion::Float64
    iterations::Int
    inner_iterations::Int
    elapsed::Float64
end

function Base.show(io::IO, r::KiteResult{M}) where {M}
    print(io, "KiteResult{", nameof(M), "}(", r.baseline.N, "×", r.baseline.J, ", ",
          r.converged ? "converged" : "NOT converged", " in ", r.iterations, " iterations)")
end

function Base.show(io::IO, ::MIME"text/plain", r::KiteResult{M}) where {M}
    b = r.baseline
    println(io, "KiteResult{", nameof(M), "}: \"", r.scenario.label, "\"")
    println(io, "  ", b.N, " countries × ", b.J, " sectors")
    @printf(io, "  %s after %d iterations (criterion %.2e, %.3g s)\n",
            r.converged ? "converged" : "DID NOT CONVERGE", r.iterations, r.criterion, r.elapsed)
    w = welfare_change(r)
    ord = sortperm(w)
    println(io, "  welfare change (Î/P̂):")
    for i in Iterators.reverse(ord[max(end - 2, 1):end])
        @printf(io, "    %-6s %+7.3f%%\n", b.countries[i], 100 * (w[i] - 1))
    end
    println(io, "    …")
    for i in ord[1:min(3, length(ord))]
        @printf(io, "    %-6s %+7.3f%%\n", b.countries[i], 100 * (w[i] - 1))
    end
    print(io, "  use `results(r; level = :country)` for the full table")
end
