# Felbermayr, Hinz, Krantz, Mahlkow & Wanner (2025), "Regional impact of global shocks".
#
# A Baqaee & Farhi (2024) network trade model with regions, extended with labour mobility after
# Caliendo, Dvorkin & Parro (2019). Relative to Caliendo & Parro (2015) it adds four things:
#
#   * **Nested CES throughout** rather than Cobb-Douglas — σ across sectors in consumption, ρ
#     between value added and intermediates, κ between labour and structures, η across
#     intermediate sectors. Every cost and expenditure share is therefore endogenous and moves
#     with relative prices. At unit elasticities the nests collapse to Cobb-Douglas, shares stop
#     moving, and the model reduces to Caliendo & Parro with two factors.
#
#   * **Two factors.** Labour is mobile within a group of regions; local structures are in fixed
#     supply and earn a rent. Rents are collected into a *global portfolio* and paid out in fixed
#     shares `ι`, so a region's income is not tied to the capital located in it.
#
#   * **Labour mobility.** Workers reallocate across regions of the same group until real wages
#     are equalised, which makes the regional labour force endogenous.
#
#   * **Tariff-revenue pooling.** Regions that are countries keep their own tariff revenue;
#     regions inside a customs union pool it and redistribute by fixed shares.
#
# Sourcing is use-specific as in Antràs & Chor (2018): π[o,d,j,k] for sector-j goods bought by
# sector k, plus π[o,d,j,C] for final consumption.
#
# There is no separate trade balance. Income is labour income plus the portfolio share of world
# rents plus tariff revenue, and `ι` absorbs whatever imbalance the data contains — the role `D`
# plays in `CaliendoParro2015`.

"""
    FelbermayrEtAl2025(; σ = 1.0, ρ = 0.6, κ = 1.0, η = 0.2, mobility = :mobile)

Regional trade model with labour mobility, after Felbermayr, Hinz, Krantz, Mahlkow & Wanner
(2025), building on Baqaee & Farhi (2024) and Caliendo, Dvorkin & Parro (2019).

Requires a [`RegionalBaseline`](@ref).

# Keyword Arguments
- `σ`: elasticity of substitution across sectors in final consumption.
- `ρ`: between value added and intermediates in production.
- `κ`: between labour and local structures.
- `η`: across sectoral intermediate bundles.
- `mobility`: `:mobile` lets workers reallocate across regions of the same group until real
  wages are equalised; `:immobile` fixes each region's labour force. Groups are set on the
  baseline — see [`RegionalBaseline`](@ref).

Any elasticity equal to one gives the Cobb-Douglas nest, in which the corresponding shares are
constant. With **all four** at one and `mobility = :immobile`, the model reduces to
[`CaliendoParro2015`](@ref) with the two factors merged — which is how it is tested.

# Examples
```julia
m = FelbermayrEtAl2025(; σ = 1.0, ρ = 0.6, κ = 1.0, η = 0.2, mobility = :mobile)
r = update_equilibrium(m, rb, sc)
labour_reallocation(r)
```
"""
struct FelbermayrEtAl2025 <: KiteModel
    σ::Float64
    ρ::Float64
    κ::Float64
    η::Float64
    mobility::Symbol

    function FelbermayrEtAl2025(; σ::Real = 1.0, ρ::Real = 0.6, κ::Real = 1.0,
                                  η::Real = 0.2, mobility::Symbol = :mobile)
        mobility in (:mobile, :immobile) ||
            error("mobility must be :mobile or :immobile; got :$mobility.")
        for (name, e) in (("σ", σ), ("ρ", ρ), ("κ", κ), ("η", η))
            e > 0 || error("$name must be positive; got $e.")
        end
        return new(Float64(σ), Float64(ρ), Float64(κ), Float64(η), mobility)
    end
end

_model_name(::FelbermayrEtAl2025) = "felbermayr_et_al_2025"

Base.show(io::IO, m::FelbermayrEtAl2025) =
    print(io, "FelbermayrEtAl2025(σ = ", m.σ, ", ρ = ", m.ρ, ", κ = ", m.κ, ", η = ", m.η,
          ", mobility = :", m.mobility, ")")

# ── CES helpers ───────────────────────────────────────────────────────────────────────────
#
# The Cobb-Douglas case e == 1 is a removable singularity of the CES index: as e → 1 the
# exponent 1/(1-e) diverges but the bracket tends to 1 at the same rate, and the limit is the
# geometric mean. It has to be special-cased or the formula returns NaN.

_is_unit(e) = abs(e - 1) < 1e-9

"""
    _ces2(s1, x1, s2, x2, e)

Two-input CES price index `(s1·x1^(1-e) + s2·x2^(1-e))^(1/(1-e))`, with the Cobb-Douglas limit
`x1^s1 · x2^s2` at `e == 1`. Shares are assumed to sum to one.
"""
@inline function _ces2(s1, x1, s2, x2, e)
    if _is_unit(e)
        return exp(s1 * log(x1) + s2 * log(x2))
    end
    p = 1 - e
    return (s1 * x1^p + s2 * x2^p)^(1 / p)
end

"""
    _share_update(s, x, index, e)

CES share revision `s·(x/index)^(1-e)`, which is the identity at `e == 1` (Cobb-Douglas shares
are constant).
"""
@inline _share_update(s, x, index, e) = _is_unit(e) ? s : s * (x / index)^(1 - e)

"""
    RegionalBaseline

Baseline for [`FelbermayrEtAl2025`](@ref): a regionalized input-output economy with two factors,
use-specific sourcing, mobility groups and tariff pools.

Consistency, enforced by the constructor, is the counterpart of [`KiteBaseline`](@ref)'s:

    Y[o,j] = Σ_d π_fin[o,d,j]/(τζ)·α[d,j]·I[d]
           + Σ_{k,d} π_use[o,d,j,k]/(τζ)·βM[d,k]·γ[d,j,k]·Y[d,k]      goods market
    wL[o]  = Σ_j βF[o,j]·γL[o,j]·Y[o,j]                                labour market
    rS[o]  = Σ_j βF[o,j]·(1−γL[o,j])·Y[o,j]                            structures market
    I[d]   = wL[d] + ι[d]·Σ_o rS[o] + T[d]                             income

# Fields
- `N`, `J`, `regions`, `sectors`, `region_index`, `sector_index`
- `group::Vector{Int}`: mobility group per region; workers move freely within a group and not
  across. `group_labels` names them. A region in a group of its own is effectively immobile.
- `pool::Vector{Int}`: tariff-revenue pool per region; `0` means the region keeps its own.
  `ζ::Vector{Float64}` gives each pooled region's share of its pool's revenue.
- `π_use::Array{Float64,4}` `(N,N,J,J)`, `π_fin::Array{Float64,3}` `(N,N,J)`: sourcing shares.
- `βF::Matrix{Float64}` `(N,J)`: value-added share of gross output (`βM = 1 − βF`).
- `γL::Matrix{Float64}` `(N,J)`: labour share of value added (`γS = 1 − γL`).
- `γ::Array{Float64,3}` `(N,J,J)` `[d,k,j]`: composition of sector `j`'s intermediate bundle,
  summing to one over `k`.
- `α::Matrix{Float64}` `(N,J)`: final-consumption shares, summing to one.
- `θ::Vector{Float64}`, `τ::Array{Float64,3}`, `ζ_trade::Array{Float64,3}`: trade elasticities
  and product-specific policy.
- `Y`, `I`, `L`, `wL`, `rS`, `T`, `ι`: gross output, income, labour force, labour income,
  structure rents, tariff revenue, and global-portfolio shares.
"""
struct RegionalBaseline
    N::Int
    J::Int
    regions::Vector{String}
    sectors::Vector{String}
    region_index::Dict{String,Int}
    sector_index::Dict{String,Int}
    group::Vector{Int}
    group_labels::Vector{String}
    pool::Vector{Int}
    ζ::Vector{Float64}
    π_use::Array{Float64,4}
    π_fin::Array{Float64,3}
    βF::Matrix{Float64}
    γL::Matrix{Float64}
    γ::Array{Float64,3}
    α::Matrix{Float64}
    θ::Vector{Float64}
    τ::Array{Float64,3}
    ζ_trade::Array{Float64,3}
    Y::Matrix{Float64}
    I::Vector{Float64}
    L::Vector{Float64}
    wL::Vector{Float64}
    rS::Vector{Float64}
    T::Vector{Float64}
    ι::Vector{Float64}
end

function Base.show(io::IO, rb::RegionalBaseline)
    print(io, "RegionalBaseline(", rb.N, " regions × ", rb.J, " sectors, ",
          length(unique(rb.group)), " mobility groups)")
end

function Base.show(io::IO, ::MIME"text/plain", rb::RegionalBaseline)
    res = residuals(rb)
    println(io, "RegionalBaseline: ", rb.N, " regions × ", rb.J, " sectors")
    println(io, "  regions  : ", _preview(rb.regions))
    println(io, "  sectors  : ", _preview(rb.sectors))
    println(io, "  mobility groups: ", length(unique(rb.group)),
            " (largest ", maximum(count(==(g), rb.group) for g in unique(rb.group)), " regions)")
    println(io, "  tariff pools   : ", length(unique(filter(>(0), rb.pool))))
    @printf(io, "  labour income %.4g, structure rents %.4g, tariff revenue %.4g\n",
            sum(rb.wL), sum(rb.rS), sum(rb.T))
    @printf(io, "  residuals: goods %.2e, labour %.2e, structures %.2e, income %.2e",
            res.goods_market, res.labour, res.structures, res.income)
end

"""
    residuals(rb::RegionalBaseline) -> NamedTuple

Maximum relative violation of the four baseline identities: `goods_market`, `labour`,
`structures` and `income`.
"""
function residuals(rb::RegionalBaseline)
    N, J = rb.N, rb.J
    F = rb.α .* rb.I
    M = Array{Float64,3}(undef, N, J, J)
    @inbounds for k in 1:J, j in 1:J
        @views @. M[:, j, k] = (1 - rb.βF[:, k]) * rb.γ[:, j, k] * rb.Y[:, k]
    end
    Y = similar(rb.Y)
    _use_output!(Y, rb.π_use, rb.π_fin, M, F, rb.τ, rb.ζ_trade)
    goods = _max_rel_dev(Y, rb.Y)

    wL = vec(sum(rb.βF .* rb.γL .* rb.Y, dims = 2))
    rS = vec(sum(rb.βF .* (1 .- rb.γL) .* rb.Y, dims = 2))
    lab = _max_rel_dev(wL, rb.wL)
    str = _max_rel_dev(rS, rb.rS)
    inc = _max_rel_dev(rb.wL .+ rb.ι .* sum(rb.rS) .+ rb.T, rb.I)
    return (goods_market = goods, labour = lab, structures = str, income = inc)
end

# ── regional-specific kernels ─────────────────────────────────────────────────────────────
# The use-specific sourcing kernels this model shares with `AntrasChor2018` live in
# `use_specific.jl`; what follows is particular to the regional model.

"""
    _reg_tariff_revenue(π_use, π_fin, M, F, τ, ζ) -> Vector

Tariff revenue collected at each destination before any pooling,
`Σ_{o,j} (τ−1)/τ · ( π_fin·F + Σ_k π_use·M )`.
"""
function _reg_tariff_revenue(π_use, π_fin, M, F, τ, ζ)
    N, J = size(F)
    T = zeros(N)
    @inbounds for j in 1:J, d in 1:N
        acc = 0.0
        f = F[d, j]
        for o in 1:N
            w = (τ[o, d, j] - 1) / τ[o, d, j]
            w == 0 && continue
            s = π_fin[o, d, j] * f
            for k in 1:J
                s += π_use[o, d, j, k] * M[d, j, k]
            end
            acc += w * s
        end
        T[d] += acc
    end
    return T
end

"""
    _apply_pooling(T_raw, pool, ζ) -> Vector

Redistribute tariff revenue. A region with `pool == 0` keeps what it collected; regions sharing
a positive pool id hand their revenue to the pool, which pays out in proportion to `ζ`.
"""
function _apply_pooling(T_raw, pool, ζ)
    T = copy(T_raw)
    for p in unique(pool)
        p == 0 && continue
        idx = findall(==(p), pool)
        total = sum(@view T_raw[idx])
        shares = @view ζ[idx]
        s = sum(shares)
        s > 0 || error("tariff pool $p has ζ shares summing to zero.")
        @inbounds for (i, d) in enumerate(idx)
            T[d] = total * shares[i] / s
        end
    end
    return T
end

# ── calibration ───────────────────────────────────────────────────────────────────────────

"""
    calibrate_regional(; regions, sectors, π_use, π_fin, βF, γL, γ, θ, F, L,
                         τ = nothing, ζ_trade = nothing, group = nothing, pool = nothing,
                         ζ = nothing, verbose = 1) -> RegionalBaseline

Build a consistent [`RegionalBaseline`](@ref) from shares and final-demand levels.

Final demand `F` is the anchor, as in [`calibrate`](@ref): consumption shares and income follow
from it directly, `α = F / Σ_j F` and `I = Σ_j F`. Gross output then solves the goods-market
condition, which given `F` is a linear fixed point in `Y`,

    Y[o,j] = Σ_d π_fin[o,d,j]·F[d,j]/(τζ) + Σ_{k,d} π_use[o,d,j,k]·βM[d,k]·γ[d,j,k]·Y[d,k]/(τζ),

and converges because value added leaks out of the intermediate loop at every round. Factor
incomes follow from the cost shares, tariff revenue from the policy wedges, and the
global-portfolio shares `ι` are the residual that closes the income identity — the role the
trade balance plays in [`calibrate`](@ref).

# Keyword Arguments
- `regions`, `sectors`: label vectors of length `N` and `J`.
- `π_use` `(N,N,J,J)`, `π_fin` `(N,N,J)`: sourcing shares; renormalised over origins.
- `βF` `(N,J)`: value-added share of gross output. `γL` `(N,J)`: labour share of value added.
- `γ` `(N,J,J)` `[d,k,j]`: intermediate-bundle composition, renormalised over `k`.
- `F` `(N,J)`: final-demand levels. `L` `(N)`: labour force.
- `θ` `(J)`, `τ`, `ζ_trade` `(N,N,J)`: elasticities and product-specific policy.
- `group`: mobility group per region, as labels or integers. Defaults to one group per region,
  which is the immobile case.
- `pool`, `ζ`: tariff pool per region and redistribution shares within it. By default every
  region keeps the revenue it collects.
"""
function calibrate_regional(; regions, sectors, π_use, π_fin, βF, γL, γ, θ, F, L,
                              τ = nothing, ζ_trade = nothing,
                              group = nothing, pool = nothing, ζ = nothing,
                              clip::Float64 = 1e-6, tolerance::Float64 = 1e-15,
                              max_iterations::Int = 100_000, verbose::Int = 1)
    N, J = length(regions), length(sectors)
    regions = String.(regions); sectors = String.(sectors)
    _check_size(π_use, (N, N, J, J), "π_use")
    _check_size(π_fin, (N, N, J), "π_fin")
    _check_size(βF, (N, J), "βF"); _check_size(γL, (N, J), "γL")
    _check_size(γ, (N, J, J), "γ"); _check_size(F, (N, J), "F")
    _check_size(L, (N,), "L"); _check_size(θ, (J,), "θ")

    pu = Array{Float64,4}(π_use); pf = Array{Float64,3}(π_fin)
    _normalise_use_shares!(pu); _normalise_trade_shares!(pf; verbose = 0)
    βF = clamp.(Float64.(βF), clip, 1 - clip)
    γL = clamp.(Float64.(γL), clip, 1 - clip)
    γ = Array{Float64,3}(γ); _normalise_intermediate_shares!(γ)
    F = max.(Matrix{Float64}(F), 0.0)
    L = Vector{Float64}(L); θ = Vector{Float64}(θ)
    τ = τ === nothing ? ones(N, N, J) : Array{Float64,3}(τ)
    ζ_trade = ζ_trade === nothing ? ones(N, N, J) : Array{Float64,3}(ζ_trade)
    all(>(0), L) || error("calibrate_regional: labour force must be positive everywhere.")

    grp, grp_labels = _resolve_groups(group, regions)
    pl = pool === nothing ? zeros(Int, N) : _resolve_pools(pool, regions)
    zz = ζ === nothing ? ones(N) : Vector{Float64}(ζ)
    _check_size(zz, (N,), "ζ")

    I = vec(sum(F, dims = 2))
    all(>(0), I) || error("calibrate_regional: every region needs positive final demand.")
    α = F ./ I

    # gross output: linear fixed point given final demand
    Y = copy(F)
    Ynew = similar(Y)
    M = Array{Float64,3}(undef, N, J, J)
    passes = 0
    for it in 1:max_iterations
        passes = it
        @inbounds for k in 1:J, j in 1:J
            @views @. M[:, j, k] = (1 - βF[:, k]) * γ[:, j, k] * Y[:, k]
        end
        _use_output!(Ynew, pu, pf, M, F, τ, ζ_trade)
        dev = _max_rel_dev(Ynew, Y)
        copyto!(Y, Ynew)
        dev < tolerance && break
    end
    @inbounds for k in 1:J, j in 1:J
        @views @. M[:, j, k] = (1 - βF[:, k]) * γ[:, j, k] * Y[:, k]
    end

    wL = vec(sum(βF .* γL .* Y, dims = 2))
    rS = vec(sum(βF .* (1 .- γL) .* Y, dims = 2))
    all(>(0), wL) || error("calibrate_regional: some region has no labour income.")

    T = _apply_pooling(_reg_tariff_revenue(pu, pf, M, F, τ, ζ_trade), pl, zz)
    total_rents = sum(rS)
    ι = total_rents > 0 ? (I .- wL .- T) ./ total_rents : zeros(N)

    rb = RegionalBaseline(N, J, regions, sectors,
                          Dict(r => i for (i, r) in enumerate(regions)),
                          Dict(s => i for (i, s) in enumerate(sectors)),
                          grp, grp_labels, pl, zz, pu, pf, βF, γL, γ, α, θ, τ, ζ_trade,
                          Y, I, L, wL, rS, T, ι)

    if verbose >= 1
        res = residuals(rb)
        @info @sprintf("calibrate_regional: %d passes to consistent output; residuals (goods \
                        %.1e, labour %.1e, structures %.1e, income %.1e); Σι = %.4f",
                       passes, res.goods_market, res.labour, res.structures, res.income,
                       sum(ι))
    end
    return rb
end

function _resolve_groups(group, regions)
    N = length(regions)
    group === nothing && return collect(1:N), copy(regions)
    length(group) == N || throw(DimensionMismatch("group has length $(length(group)), expected $N."))
    labels = unique(String.(group))
    idx = Dict(l => i for (i, l) in enumerate(labels))
    return [idx[String(g)] for g in group], labels
end

function _resolve_pools(pool, regions)
    N = length(regions)
    length(pool) == N || throw(DimensionMismatch("pool has length $(length(pool)), expected $N."))
    eltype(pool) <: Integer && return collect(Int, pool)
    labels = unique(String[String(p) for p in pool if String(p) != ""])
    idx = Dict(l => i for (i, l) in enumerate(labels))
    return [String(p) == "" ? 0 : idx[String(p)] for p in pool]
end

# ── workspace ─────────────────────────────────────────────────────────────────────────────

struct _RegWorkspace
    N::Int
    J::Int
    ŵ::Vector{Float64}
    ŵ_prev::Vector{Float64}
    r̂::Vector{Float64}
    L̂::Vector{Float64}
    P̂C::Vector{Float64}
    I′::Vector{Float64}
    T′::Vector{Float64}
    Ψ::Vector{Float64}
    labour′::Vector{Float64}
    rent′::Vector{Float64}
    u::Vector{Float64}
    t::Vector{Float64}
    ĉ::Matrix{Float64}
    P̂F::Matrix{Float64}
    P̂M::Matrix{Float64}
    P̂fin::Matrix{Float64}
    P̂use::Array{Float64,3}          # (N,J,J) [d,j,k]: sector-j goods bought by sector k
    P̂use_prev::Array{Float64,3}
    βF′::Matrix{Float64}
    γL′::Matrix{Float64}
    γ′::Array{Float64,3}
    α′::Matrix{Float64}
    Y′::Matrix{Float64}
    Y_prev::Matrix{Float64}
    F′::Matrix{Float64}
    M′::Array{Float64,3}
    tr_fin::Matrix{Float64}
    tr_use::Array{Float64,3}
    φ̂::Array{Float64,3}
    W_fin::Array{Float64,3}
    W_use::Array{Float64,4}
    A_fin::Array{Float64,3}
    A_use::Array{Float64,4}
    has_productivity::Bool
    has_tariff::Bool
end

function _RegWorkspace(rb::RegionalBaseline, sc::Scenario)
    N, J = rb.N, rb.J
    φ̂ = similar(rb.τ)
    @inbounds @. φ̂ = (sc.τ′ / rb.τ) * sc.κ̂ * (sc.ζ′ / rb.ζ_trade)

    W_fin = similar(rb.τ)
    @inbounds for j in 1:J
        θj = rb.θ[j]
        @views @. W_fin[:, :, j] = rb.π_fin[:, :, j] * φ̂[:, :, j]^(-θj)
    end
    W_use = Array{Float64,4}(undef, N, N, J, J)
    @inbounds for k in 1:J, j in 1:J
        θj = rb.θ[j]
        @views @. W_use[:, :, j, k] = rb.π_use[:, :, j, k] * φ̂[:, :, j]^(-θj)
    end

    return _RegWorkspace(N, J,
        ones(N), ones(N), ones(N), ones(N), ones(N), copy(rb.I), copy(rb.T), ones(N),
        copy(rb.wL), copy(rb.rS), zeros(N), zeros(N),
        ones(N, J), ones(N, J), ones(N, J), ones(N, J),
        ones(N, J, J), ones(N, J, J),
        copy(rb.βF), copy(rb.γL), copy(rb.γ), copy(rb.α),
        copy(rb.Y), similar(rb.Y), rb.α .* rb.I, Array{Float64,3}(undef, N, J, J),
        zeros(N, J), zeros(N, J, J),
        φ̂, W_fin, W_use, similar(rb.τ), Array{Float64,4}(undef, N, N, J, J),
        any(!=(1.0), sc.ẑ), any(!=(1.0), sc.τ′))
end

# ── result ────────────────────────────────────────────────────────────────────────────────

"""
    RegionalResult

Solved counterfactual for [`FelbermayrEtAl2025`](@ref).

# Fields
- `model`, `baseline::RegionalBaseline`, `scenario`, `settings`
- `ŵ`, `r̂`, `L̂`, `P̂C`: wage, structure rental price, labour force and consumer price index.
- `ĉ`, `P̂fin`, `P̂use`: unit costs and price indices (`P̂use` is `(N,J,J)`).
- `Y′`, `F′`, `M′`, `I′`, `T′`: output, final demand, intermediate demand, income and tariff
  revenue.
- `βF′`, `γL′`, `α′`: revised cost and expenditure shares (constant at unit elasticities).
- `π_use′`, `π_fin′`, `π̄′`: counterfactual sourcing shares and their expenditure-weighted mean.
- `converged`, `criterion`, `iterations`, `inner_iterations`, `elapsed`

Real income per worker is the model's welfare measure — see [`welfare_change`](@ref) and
[`labour_reallocation`](@ref).
"""
struct RegionalResult
    model::FelbermayrEtAl2025
    baseline::RegionalBaseline
    scenario::Scenario
    settings::SolverSettings
    ŵ::Vector{Float64}
    r̂::Vector{Float64}
    L̂::Vector{Float64}
    P̂C::Vector{Float64}
    ĉ::Matrix{Float64}
    P̂fin::Matrix{Float64}
    P̂use::Array{Float64,3}
    Y′::Matrix{Float64}
    F′::Matrix{Float64}
    M′::Array{Float64,3}
    I′::Vector{Float64}
    T′::Vector{Float64}
    βF′::Matrix{Float64}
    γL′::Matrix{Float64}
    α′::Matrix{Float64}
    π_use′::Array{Float64,4}
    π_fin′::Array{Float64,3}
    π̄′::Array{Float64,3}
    converged::Bool
    criterion::Float64
    iterations::Int
    inner_iterations::Int
    elapsed::Float64
end

function Base.show(io::IO, r::RegionalResult)
    print(io, "RegionalResult(", r.baseline.N, "×", r.baseline.J, ", ",
          r.converged ? "converged" : "NOT converged", " in ", r.iterations, " iterations)")
end

function Base.show(io::IO, ::MIME"text/plain", r::RegionalResult)
    rb = r.baseline
    println(io, "RegionalResult: \"", r.scenario.label, "\"  [", r.model.mobility, "]")
    println(io, "  ", rb.N, " regions × ", rb.J, " sectors")
    @printf(io, "  %s after %d iterations (criterion %.2e, %.3g s)\n",
            r.converged ? "converged" : "DID NOT CONVERGE", r.iterations, r.criterion,
            r.elapsed)
    w = welfare_change(r)
    @printf(io, "  real income per worker ∈ [%.4f, %.4f]\n", extrema(w)...)
    if r.model.mobility === :mobile
        @printf(io, "  labour reallocation L̂ ∈ [%.4f, %.4f]\n", extrema(r.L̂)...)
    end
    print(io, "  use `results(r)` for the full table")
end

# ── solver ────────────────────────────────────────────────────────────────────────────────

"""
    update_equilibrium(m::FelbermayrEtAl2025, rb::RegionalBaseline, sc = Scenario(rb); kwargs...)

Solve the regional equilibrium in changes, returning a [`RegionalResult`](@ref).

The outer loop iterates on the wage `ŵ` and the structure rental price `r̂`, with the regional
labour force `L̂` endogenous when `mobility = :mobile`. Because every cost share is endogenous
under CES, the inner expenditure block is re-linearised each outer pass.

# Examples
```julia
m  = FelbermayrEtAl2025(; ρ = 0.6, η = 0.2, mobility = :mobile)
sc = Scenario(rb); set_tariff!(sc, rb, 1.25; from = :all, to = "USA")
r  = update_equilibrium(m, rb, sc)
```
"""
function update_equilibrium(m::FelbermayrEtAl2025, rb::RegionalBaseline,
                            sc::Scenario = Scenario(rb); kwargs...)
    return update_equilibrium(m, rb, sc, SolverSettings(; kwargs...))
end

function update_equilibrium(m::FelbermayrEtAl2025, rb::RegionalBaseline, sc::Scenario,
                            settings::SolverSettings)
    t0 = time()
    _reg_check_conformable(rb, sc)
    settings.inner_solver === :iterative ||
        error("inner_solver must be :iterative for felbermayr_et_al_2025; its cost shares " *
              "move with prices, so the expenditure block is not a fixed linear system.")

    ws = _RegWorkspace(rb, sc)
    num_index = settings.numeraire isa Symbol ? 0 :
        (haskey(rb.region_index, settings.numeraire) ? rb.region_index[settings.numeraire] :
         error("numeraire region \"$(settings.numeraire)\" is not in the baseline."))

    criterion = Inf
    inner_ok = true
    inner_total = 0
    iter = 0

    while iter < settings.max_iterations
        iter += 1
        copyto!(ws.ŵ_prev, ws.ŵ)
        copyto!(ws.P̂use_prev, ws.P̂use)

        _reg_cost_nests!(m, ws, rb, sc)
        _reg_prices!(ws, rb)
        _reg_policy_weights!(ws, rb, sc)
        _reg_consumption!(m, ws, rb)

        n_inner, inner_ok = _reg_inner!(ws, rb, settings)
        inner_total += n_inner

        _reg_factor_update!(m, ws, rb, settings)
        _reg_normalise!(ws, rb, num_index)

        criterion = max(_criterion(ws.ŵ, ws.ŵ_prev, settings.convergence),
                        _criterion(ws.P̂use, ws.P̂use_prev, settings.convergence))
        settings.verbose ≥ 2 &&
            @printf("  iteration %4d   criterion %.3e   inner %d\n", iter, criterion, n_inner)

        criterion ≤ settings.tolerance &&
            (!settings.require_inner_convergence || inner_ok) && break
    end

    converged = isfinite(criterion) && criterion ≤ settings.tolerance &&
                (!settings.require_inner_convergence || inner_ok)
    elapsed = time() - t0

    if settings.verbose ≥ 1
        if converged
            @info @sprintf("felbermayr_et_al_2025: converged in %d iterations (criterion \
                            %.2e, %.3g s)", iter, criterion, elapsed)
        else
            @warn @sprintf("felbermayr_et_al_2025: did NOT converge in %d iterations \
                            (criterion %.2e).", iter, criterion)
        end
    end

    π_use′, π_fin′ = _reg_recover_shares(ws, sc)
    X′ = similar(ws.Y′)
    @inbounds for j in 1:ws.J
        @views @. X′[:, j] = ws.F′[:, j]
        for k in 1:ws.J
            @views @. X′[:, j] += ws.M′[:, j, k]
        end
    end
    π̄ = _use_aggregate_shares(π_use′, π_fin′, ws.M′, ws.F′, X′)

    return RegionalResult(m, rb, sc, settings,
                          copy(ws.ŵ), copy(ws.r̂), copy(ws.L̂), copy(ws.P̂C),
                          copy(ws.ĉ), copy(ws.P̂fin), copy(ws.P̂use),
                          copy(ws.Y′), copy(ws.F′), copy(ws.M′), copy(ws.I′), copy(ws.T′),
                          copy(ws.βF′), copy(ws.γL′), copy(ws.α′),
                          π_use′, π_fin′, π̄,
                          converged, criterion, iter, inner_total, elapsed)
end

function _reg_check_conformable(rb::RegionalBaseline, sc::Scenario)
    _check_size(sc.τ′, (rb.N, rb.N, rb.J), "scenario τ′")
    _check_size(sc.ζ′, (rb.N, rb.N, rb.J), "scenario ζ′")
    _check_size(sc.κ̂, (rb.N, rb.N, rb.J), "scenario κ̂")
    _check_size(sc.ẑ, (rb.N, rb.J), "scenario ẑ")
    all(>(0), sc.τ′) || error("scenario τ′ must be strictly positive.")
    all(>(0), sc.κ̂) || error("scenario κ̂ must be strictly positive.")
    all(>(0), sc.ζ′) || error("scenario ζ′ must be strictly positive.")
    return nothing
end

"""
    Scenario(rb::RegionalBaseline; label = "scenario")

Counterfactual policy for a [`RegionalBaseline`](@ref), with the same fields and setters as the
country-level constructor. The `coalition` field is unused by this model.
"""
Scenario(rb::RegionalBaseline; label::AbstractString = "scenario") =
    Scenario(copy(rb.τ), copy(rb.ζ_trade), ones(rb.N, rb.N, rb.J), ones(rb.N, rb.J),
             ones(rb.N), falses(rb.N), String(label))

_countries(rb::RegionalBaseline, sel) = _resolve(sel, rb.region_index, rb.N, "region")
_sectors(rb::RegionalBaseline, sel) = _resolve(sel, rb.sector_index, rb.J, "sector")

"""
    _reg_cost_nests!(m, ws, rb, sc)

Paper equations (input costs, input shares, factor price index, factor shares, intermediate
price index, intermediate shares). Working outward from the factors:

    P̂F[d,j] = CES_κ( γL, ŵ ; 1−γL, r̂ )
    P̂M[d,j] = CES_η( γ[d,·,j], P̂use[d,·,j] )
    ĉ[d,j]  = CES_ρ( βF, P̂F ; 1−βF, P̂M )

with each nest's shares revised by `s·(x/index)^(1−e)`. At unit elasticity the CES index is the
geometric mean and shares do not move — the Cobb-Douglas case.
"""
function _reg_cost_nests!(m::FelbermayrEtAl2025, ws::_RegWorkspace, rb::RegionalBaseline,
                          sc::Scenario)
    N, J = ws.N, ws.J
    @inbounds for j in 1:J, d in 1:N
        gL = rb.γL[d, j]
        PF = _ces2(gL, ws.ŵ[d], 1 - gL, ws.r̂[d], m.κ)
        ws.P̂F[d, j] = PF
        ws.γL′[d, j] = _share_update(gL, ws.ŵ[d], PF, m.κ)

        PM = if _is_unit(m.η)
            acc = 0.0
            for k in 1:J
                acc += rb.γ[d, k, j] * log(ws.P̂use[d, k, j])
            end
            exp(acc)
        else
            p = 1 - m.η
            acc = 0.0
            for k in 1:J
                acc += rb.γ[d, k, j] * ws.P̂use[d, k, j]^p
            end
            acc^(1 / p)
        end
        ws.P̂M[d, j] = PM
        for k in 1:J
            ws.γ′[d, k, j] = _share_update(rb.γ[d, k, j], ws.P̂use[d, k, j], PM, m.η)
        end

        bF = rb.βF[d, j]
        c = _ces2(bF, PF, 1 - bF, PM, m.ρ)
        ws.ĉ[d, j] = c
        ws.βF′[d, j] = _share_update(bF, PF, c, m.ρ)
    end
    ws.has_productivity && @inbounds @. ws.ĉ /= sc.ẑ
    return ws
end

"""
    _reg_prices!(ws, rb)

Use-specific price indices `P̂[d,j,k] = (Σ_o π[o,d,j,k]·(φ̂·ĉ_o^j)^(−θ_j))^(−1/θ_j)` and the
final-consumption counterpart, one `gemv!` each against the cached `W = π·φ̂^(−θ)`.
"""
function _reg_prices!(ws::_RegWorkspace, rb::RegionalBaseline)
    N, J = ws.N, ws.J
    @inbounds for j in 1:J
        θj = rb.θ[j]
        @views @. ws.u = ws.ĉ[:, j]^(-θj)
        @views mul!(ws.t, transpose(view(ws.W_fin, :, :, j)), ws.u)
        @views @. ws.P̂fin[:, j] = ifelse(ws.t > 0, ws.t^(-1 / θj), 1.0)
        for k in 1:J
            @views mul!(ws.t, transpose(view(ws.W_use, :, :, j, k)), ws.u)
            @views @. ws.P̂use[:, j, k] = ifelse(ws.t > 0, ws.t^(-1 / θj), 1.0)
        end
    end
    return ws
end

"""
    _reg_policy_weights!(ws, rb, sc)

`A = π′/(τ′ζ′)` for both flows, plus the tariff-revenue weights. Since `π′ = A·τ′ζ′`, revenue
per unit of expenditure is `(τ′−1)/τ′·π′ = (τ′−1)·ζ′·A`, so the weights collapse to
`tr[d,j,·] = Σ_o (τ′−1)·ζ′·A`.
"""
function _reg_policy_weights!(ws::_RegWorkspace, rb::RegionalBaseline, sc::Scenario)
    N, J = ws.N, ws.J
    @inbounds for j in 1:J
        θj = rb.θ[j]
        @views @. ws.u = ws.ĉ[:, j]^(-θj)
        for d in 1:N
            vf = ws.P̂fin[d, j]^θj
            @views @. ws.A_fin[:, d, j] =
                ws.W_fin[:, d, j] * ws.u * vf / (sc.τ′[:, d, j] * sc.ζ′[:, d, j])
            for k in 1:J
                vk = ws.P̂use[d, j, k]^θj
                @views @. ws.A_use[:, d, j, k] =
                    ws.W_use[:, d, j, k] * ws.u * vk / (sc.τ′[:, d, j] * sc.ζ′[:, d, j])
            end
        end
    end

    if ws.has_tariff
        @inbounds for j in 1:J, d in 1:N
            acc = 0.0
            @simd for o in 1:N
                acc += (sc.τ′[o, d, j] - 1) * sc.ζ′[o, d, j] * ws.A_fin[o, d, j]
            end
            ws.tr_fin[d, j] = acc
            for k in 1:J
                a = 0.0
                @simd for o in 1:N
                    a += (sc.τ′[o, d, j] - 1) * sc.ζ′[o, d, j] * ws.A_use[o, d, j, k]
                end
                ws.tr_use[d, j, k] = a
            end
        end
    else
        fill!(ws.tr_fin, 0.0)
        fill!(ws.tr_use, 0.0)
    end
    return ws
end

"""
    _reg_consumption!(m, ws, rb)

Consumer price index and consumption shares: `P̂C[d] = CES_σ( α[d,·], P̂fin[d,·] )` and
`α′ = α·(P̂fin/P̂C)^(1−σ)`.
"""
function _reg_consumption!(m::FelbermayrEtAl2025, ws::_RegWorkspace, rb::RegionalBaseline)
    N, J = ws.N, ws.J
    @inbounds for d in 1:N
        PC = if _is_unit(m.σ)
            acc = 0.0
            for j in 1:J
                acc += rb.α[d, j] * log(ws.P̂fin[d, j])
            end
            exp(acc)
        else
            p = 1 - m.σ
            acc = 0.0
            for j in 1:J
                acc += rb.α[d, j] * ws.P̂fin[d, j]^p
            end
            acc^(1 / p)
        end
        ws.P̂C[d] = PC
        for j in 1:J
            ws.α′[d, j] = _share_update(rb.α[d, j], ws.P̂fin[d, j], PC, m.σ)
        end
    end
    return ws
end

"""
    _reg_inner!(ws, rb, settings) -> (iterations, converged)

Fixed point in `(Y′, I′)` holding the factor prices fixed:

    F′[d,j]   = α′[d,j]·I′[d]
    M′[d,j,k] = (1−βF′[d,k])·γ′[d,j,k]·Y′[d,k]
    Y′[o,j]   = Σ_d ( A_fin[o,d,j]·F′[d,j] + Σ_k A_use[o,d,j,k]·M′[d,j,k] )
    T′        = tariff revenue, pooled where a customs union applies
    I′[d]     = ŵ[d]·L̂[d]·wL[d] + ι[d]·Σ_o r̂[o]·rS[o] + T′[d]
"""
function _reg_inner!(ws::_RegWorkspace, rb::RegionalBaseline, settings::SolverSettings)
    N, J = ws.N, ws.J
    converged = false
    it = 0

    rent_income = 0.0
    @inbounds for o in 1:N
        rent_income += ws.r̂[o] * rb.rS[o]
    end
    labour_income = Vector{Float64}(undef, N)
    @inbounds for d in 1:N
        labour_income[d] = ws.ŵ[d] * ws.L̂[d] * rb.wL[d]
    end
    T_raw = Vector{Float64}(undef, N)

    while it < settings.max_inner_iterations
        it += 1
        copyto!(ws.Y_prev, ws.Y′)

        @inbounds for j in 1:J
            @views @. ws.F′[:, j] = ws.α′[:, j] * ws.I′
        end
        @inbounds for k in 1:J, j in 1:J
            @views @. ws.M′[:, j, k] = (1 - ws.βF′[:, k]) * ws.γ′[:, j, k] * ws.Y′[:, k]
        end

        fill!(ws.Y′, 0.0)
        @inbounds for j in 1:J
            @views mul!(ws.t, view(ws.A_fin, :, :, j), ws.F′[:, j])
            @views @. ws.Y′[:, j] += ws.t
            for k in 1:J
                @views mul!(ws.t, view(ws.A_use, :, :, j, k), ws.M′[:, j, k])
                @views @. ws.Y′[:, j] += ws.t
            end
        end

        @inbounds for d in 1:N
            acc = 0.0
            for j in 1:J
                acc += ws.tr_fin[d, j] * ws.F′[d, j]
                @simd for k in 1:J
                    acc += ws.tr_use[d, j, k] * ws.M′[d, j, k]
                end
            end
            T_raw[d] = acc
        end
        ws.T′ .= _apply_pooling(T_raw, rb.pool, rb.ζ)

        @inbounds for d in 1:N
            ws.I′[d] = labour_income[d] + rb.ι[d] * rent_income + ws.T′[d]
        end

        crit = _criterion(ws.Y′, ws.Y_prev, settings.convergence)
        isfinite(crit) || break
        if crit ≤ settings.inner_tolerance
            converged = true
            break
        end
    end
    return it, converged
end

"""
    _reg_factor_update!(m, ws, rb, settings)

Labour force, wage and rental price.

Mobility (paper's labour-force equation). Writing each region's real labour-income growth as

    Ψ_o = ( Σ_j βF′[o,j]·γL′[o,j]·Y′[o,j] ) / ( wL[o] · P̂C[o] ),

workers reallocate within a group in proportion to it,
`L̂_o = Ψ_o / Σ_{d ∈ group(o)} (L_d/L_group)·Ψ_d`. This conserves each group's labour force and
equalises real wages across it: dividing the wage equation by `P̂C` gives
`ŵ_o/P̂C_o = Ψ_o/L̂_o`, which is the same for every region of the group.

Wage and rental price then clear the two factor markets:
`ŵ_o = (Σ_j βF′·γL′·Y′)/(L̂_o·wL_o)` and `r̂_o = (Σ_j βF′·(1−γL′)·Y′)/rS_o`, both dampened.
"""
function _reg_factor_update!(m::FelbermayrEtAl2025, ws::_RegWorkspace, rb::RegionalBaseline,
                             settings::SolverSettings)
    N, J = ws.N, ws.J
    v = settings.vfactor

    fill!(ws.labour′, 0.0)
    fill!(ws.rent′, 0.0)
    @inbounds for j in 1:J
        @views @. ws.labour′ += ws.βF′[:, j] * ws.γL′[:, j] * ws.Y′[:, j]
        @views @. ws.rent′ += ws.βF′[:, j] * (1 - ws.γL′[:, j]) * ws.Y′[:, j]
    end

    if m.mobility === :mobile
        @inbounds for o in 1:N
            ws.Ψ[o] = ws.labour′[o] / (rb.wL[o] * ws.P̂C[o])
        end
        # group aggregate Σ_{d∈g} (L_d/L_g)·Ψ_d
        ngroups = maximum(rb.group)
        num = zeros(ngroups)
        den = zeros(ngroups)
        @inbounds for d in 1:N
            g = rb.group[d]
            num[g] += rb.L[d] * ws.Ψ[d]
            den[g] += rb.L[d]
        end
        @inbounds for o in 1:N
            g = rb.group[o]
            avg = num[g] / den[g]
            avg > 0 || continue
            target = ws.Ψ[o] / avg
            target > 0 || continue
            ws.L̂[o] = exp((1 - v) * log(ws.L̂[o]) + v * log(target))
        end
        # renormalise so each group's labour force is exactly conserved
        fill!(num, 0.0)
        @inbounds for d in 1:N
            num[rb.group[d]] += rb.L[d] * ws.L̂[d]
        end
        @inbounds for o in 1:N
            g = rb.group[o]
            num[g] > 0 && (ws.L̂[o] *= den[g] / num[g])
        end
    end

    @inbounds for o in 1:N
        rb.wL[o] > 0 || continue
        target = ws.labour′[o] / (ws.L̂[o] * rb.wL[o])
        target > 0 || continue
        ws.ŵ[o] = exp((1 - v) * log(ws.ŵ[o]) + v * log(target))
    end
    @inbounds for o in 1:N
        rb.rS[o] > 0 || continue
        target = ws.rent′[o] / rb.rS[o]
        target > 0 || continue
        ws.r̂[o] = exp((1 - v) * log(ws.r̂[o]) + v * log(target))
    end
    return ws
end

"""
    _reg_normalise!(ws, rb, num_index)

Impose the numéraire on both factor prices: `num_index == 0` holds world factor income fixed,
otherwise it fixes one region's wage.
"""
function _reg_normalise!(ws::_RegWorkspace, rb::RegionalBaseline, num_index::Int)
    scale = if num_index == 0
        base = sum(rb.wL) + sum(rb.rS)
        new = 0.0
        @inbounds for o in 1:ws.N
            new += ws.ŵ[o] * ws.L̂[o] * rb.wL[o] + ws.r̂[o] * rb.rS[o]
        end
        new > 0 ? base / new : 1.0
    else
        ws.ŵ[num_index] > 0 ? 1 / ws.ŵ[num_index] : 1.0
    end
    isfinite(scale) && scale > 0 || return ws
    @inbounds @. ws.ŵ *= scale
    @inbounds @. ws.r̂ *= scale
    return ws
end

function _reg_recover_shares(ws::_RegWorkspace, sc::Scenario)
    N, J = ws.N, ws.J
    π_fin′ = similar(ws.A_fin)
    π_use′ = similar(ws.A_use)
    @inbounds for j in 1:J, d in 1:N
        @views @. π_fin′[:, d, j] = ws.A_fin[:, d, j] * sc.τ′[:, d, j] * sc.ζ′[:, d, j]
        for k in 1:J
            @views @. π_use′[:, d, j, k] =
                ws.A_use[:, d, j, k] * sc.τ′[:, d, j] * sc.ζ′[:, d, j]
        end
    end
    return π_use′, π_fin′
end

# ── results ───────────────────────────────────────────────────────────────────────────────

"""
    welfare_change(r::RegionalResult) -> Vector{Float64}

Real income per worker, `(Î_d / L̂_d) / P̂C_d`.

With mobile labour a region's total income can rise simply because workers moved in, so per
capita is the meaningful measure — and it is what the paper reports (real value added per
capita). Under `mobility = :mobile` this is equalised across regions of a group up to the
non-labour components of income.
"""
welfare_change(r::RegionalResult) =
    (r.I′ ./ r.baseline.I) ./ r.L̂ ./ r.P̂C

"""
    real_wage_change(r::RegionalResult) -> Vector{Float64}

`ŵ / P̂C`. Equalised within each mobility group when `mobility = :mobile`.
"""
real_wage_change(r::RegionalResult) = r.ŵ ./ r.P̂C

"""
    labour_reallocation(r::RegionalResult) -> DataFrame

Where workers move. Columns: `region`, `group`, `labour`, `labour_new`, `labour_change`,
`real_wage_change`, `welfare_change`.

Each mobility group's total labour force is conserved by construction, so `labour_change` is a
pure reallocation within groups.
"""
function labour_reallocation(r::RegionalResult)
    rb = r.baseline
    return DataFrame(
        region = rb.regions,
        group = String[rb.group_labels[g] for g in rb.group],
        labour = rb.L,
        labour_new = rb.L .* r.L̂,
        labour_change = r.L̂,
        real_wage_change = real_wage_change(r),
        welfare_change = welfare_change(r),
    )
end

"""
    results(r::RegionalResult; level = :region) -> DataFrame

Tidy results. `:region` gives one row per region — factor prices, labour, income, output and
welfare; `:sector` one row per region-sector — prices, costs, output and final demand.
"""
function results(r::RegionalResult; level::Symbol = :region)
    rb = r.baseline
    if level === :region
        Ytot = vec(sum(rb.Y, dims = 2))
        Ytot_new = vec(sum(r.Y′, dims = 2))
        va = vec(sum(rb.βF .* rb.Y, dims = 2))
        va_new = r.ŵ .* r.L̂ .* rb.wL .+ r.r̂ .* rb.rS
        return DataFrame(
            region = rb.regions,
            group = String[rb.group_labels[g] for g in rb.group],
            wage_change = r.ŵ,
            rental_price_change = r.r̂,
            labour_change = r.L̂,
            price_index_change = r.P̂C,
            real_wage_change = real_wage_change(r),
            welfare_change = welfare_change(r),
            income = rb.I,
            income_new = r.I′,
            income_change = r.I′ ./ rb.I,
            value_added = va,
            value_added_new = va_new,
            value_added_real_change = (va_new ./ va) ./ r.P̂C,
            value_added_per_worker_real_change = (va_new ./ va) ./ r.L̂ ./ r.P̂C,
            tariff_revenue = rb.T,
            tariff_revenue_new = r.T′,
            production_total = Ytot,
            production_total_new = Ytot_new,
            production_total_change = _safe_ratio.(Ytot_new, Ytot),
            labour_force = rb.L,
            labour_force_new = rb.L .* r.L̂,
        )
    elseif level === :sector
        n = rb.N * rb.J
        region = Vector{String}(undef, n); sector = Vector{String}(undef, n)
        @inbounds for j in 1:rb.J, d in 1:rb.N
            i = d + rb.N * (j - 1)
            region[i] = rb.regions[d]; sector[i] = rb.sectors[j]
        end
        Fb = rb.α .* rb.I
        return DataFrame(
            region = region, sector = sector,
            trade_elasticity = repeat(rb.θ, inner = rb.N),
            input_cost_change = vec(r.ĉ),
            price_change = vec(r.P̂fin),
            production = vec(rb.Y), production_new = vec(r.Y′),
            production_change = _safe_ratio.(vec(r.Y′), vec(rb.Y)),
            final_demand = vec(Fb), final_demand_new = vec(r.F′),
            final_demand_change = _safe_ratio.(vec(r.F′), vec(Fb)),
            value_added_share = vec(rb.βF), value_added_share_new = vec(r.βF′),
            labour_share = vec(rb.γL), labour_share_new = vec(r.γL′),
        )
    end
    error("level must be :region or :sector; got :$level.")
end
