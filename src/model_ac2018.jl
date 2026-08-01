# Antràs & Chor (2018) global value chains. KITE whitepaper §4.1, equations (34)–(42).
#
# Caliendo & Parro assume a country sources each good from the same origins whatever it is used
# for: one trade share π[o,d,j] serves final consumption and every using sector alike. In the
# data that is plainly false — a country may import its steel for construction from one place
# and for cars from another, and its final-goods sourcing looks different again.
#
# This extension makes sourcing use-specific: π[o,d,j,k] for sector-j goods used by sector k,
# plus π[o,d,j,C] for final consumption. Each therefore has its own price index, so the input
# bundle of sector j is priced with the price indices *it* faces rather than economy-wide ones.
#
# Tariffs, NTBs and export subsidies stay product-specific — τ[o,d,j], not τ[o,d,j,k]. That
# matches how trade policy actually works (an MFN rate applies to an HS code, not to the buyer's
# industry) and keeps the [`Scenario`](@ref) type unchanged.
#
# Memory: π[o,d,j,k] is N²J². At 81 countries and 50 sectors that is 16.4M entries, 131 MB per
# array, and the solver holds two of them (plus a third with export subsidies active). The
# extension is best used on aggregations of a few dozen sectors; the whitepaper lists it as
# experimental for exactly this reason.

"""
    AntrasChor2018()

Global-value-chain extension of [`CaliendoParro2015`](@ref) after Antràs & Chor (2018), in which
sourcing patterns differ across using sectors and between intermediate and final use.

Requires a [`GVCBaseline`](@ref) rather than a plain [`KiteBaseline`](@ref). With
use-independent sourcing it solves exactly the same equilibrium as [`CaliendoParro2015`](@ref).

# Examples
```julia
g = GVCBaseline(b; π_use = π_use, π_fin = π_fin)
r = update_equilibrium(AntrasChor2018(), g, sc)
r.ext.P̂_use          # (N, J, J) price index of sector-j goods used by sector k
```
"""
struct AntrasChor2018 <: KiteModel end

_model_name(::AntrasChor2018) = "antras_chor_2018"

"""
    GVCBaseline(b::KiteBaseline; π_use = nothing, π_fin = nothing, verbose = 1)

A baseline with use-specific sourcing, for [`AntrasChor2018`](@ref).

- `π_use::Array{Float64,4}`: `(N, N, J, J)` `[o, d, j, k]`, the share of country `d`'s sector-`k`
  use of sector-`j` goods bought from `o`. Normalised to sum to one over `o`.
- `π_fin::Array{Float64,3}`: `(N, N, J)` `[o, d, j]`, the same for final consumption.

Omit both to build the use-independent case, in which every column equals the aggregate
`b.π` — useful as a reference point, since it reproduces `CaliendoParro2015` exactly.

Supplying genuine use-specific shares changes the goods-market condition, so gross output is
re-solved to be consistent with them, holding final absorption `I` fixed. Value added, policy
revenue and the trade balance follow, exactly as in [`calibrate`](@ref). The field `base` then
holds an ordinary, fully consistent [`KiteBaseline`](@ref) whose `π` is the expenditure-weighted
average of the use-specific shares — so every results helper keeps working, and running
`CaliendoParro2015` on it gives the aggregate-sourcing counterfactual to compare against.

# Fields
- `base::KiteBaseline`, `π_use::Array{Float64,4}`, `π_fin::Array{Float64,3}`
"""
struct GVCBaseline
    base::KiteBaseline
    π_use::Array{Float64,4}
    π_fin::Array{Float64,3}
end

function GVCBaseline(b::KiteBaseline; π_use = nothing, π_fin = nothing,
                     tolerance::Float64 = 1e-14, max_iterations::Int = 100_000,
                     verbose::Int = 1)
    N, J = b.N, b.J

    if π_use === nothing && π_fin === nothing
        pu = Array{Float64,4}(undef, N, N, J, J)
        @inbounds for k in 1:J
            @views pu[:, :, :, k] .= b.π
        end
        return GVCBaseline(b, pu, copy(b.π))
    end
    π_use === nothing && error("GVCBaseline: supply both π_use and π_fin, or neither.")
    π_fin === nothing && error("GVCBaseline: supply both π_use and π_fin, or neither.")
    _check_size(π_use, (N, N, J, J), "π_use")
    _check_size(π_fin, (N, N, J), "π_fin")

    pu = Array{Float64,4}(π_use)
    pf = Array{Float64,3}(π_fin)
    all(≥(0), pu) || error("π_use must be non-negative.")
    all(≥(0), pf) || error("π_fin must be non-negative.")
    _normalise_use_shares!(pu, b.π)
    _normalise_trade_shares!(pf; verbose = 0)

    # Re-solve gross output against the use-specific goods-market condition, holding final
    # absorption fixed. This is the AC analogue of `calibrate(anchor = :expenditure)`.
    Y = copy(b.Y)
    Ynew = similar(Y)
    F = b.α .* b.I
    M = Array{Float64,3}(undef, N, J, J)
    passes = 0
    for it in 1:max_iterations
        passes = it
        _gvc_intermediate_demand!(M, b.input_share, Y)
        _gvc_output!(Ynew, pu, pf, M, F, b.τ, b.ζ)
        dev = _max_rel_dev(Ynew, Y)
        copyto!(Y, Ynew)
        dev < tolerance && break
    end

    _gvc_intermediate_demand!(M, b.input_share, Y)
    X = similar(b.X)
    @inbounds for j in 1:J
        @views @. X[:, j] = F[:, j]
        for k in 1:J
            @views @. X[:, j] += M[:, j, k]
        end
    end
    π̄ = _gvc_aggregate_shares(pu, pf, M, F, X)
    VA = vec(sum(b.β .* Y, dims = 2))
    R = _revenue(π̄, X, b.τ, b.ζ)
    D = VA .+ R .- b.I

    base = KiteBaseline(b.countries, b.sectors, π̄, b.γ, b.α, b.β, b.θ, b.τ, b.ζ,
                        X, Y, b.I, R, VA, D; atol = 1e-7)

    if verbose ≥ 1
        res = residuals(base)
        @info @sprintf("GVCBaseline: %d passes to consistent output; aggregate residuals \
                        (goods %.1e, expenditure %.1e, income %.1e); Σ D = %.3g",
                       passes, res.goods_market, res.expenditure, res.income, sum(D))
    end
    return GVCBaseline(base, pu, pf)
end

Base.show(io::IO, g::GVCBaseline) =
    print(io, "GVCBaseline(", g.base.N, " countries × ", g.base.J,
          " sectors, use-specific sourcing)")

function _normalise_use_shares!(pu, π_fallback)
    N, _, J, _ = size(pu)
    @inbounds for k in 1:J, j in 1:J, d in 1:N
        s = 0.0
        @simd for o in 1:N
            s += pu[o, d, j, k]
        end
        if s > 0
            @simd for o in 1:N
                pu[o, d, j, k] /= s
            end
        else
            @simd for o in 1:N
                pu[o, d, j, k] = π_fallback[o, d, j]
            end
        end
    end
    return pu
end

"""
    _gvc_intermediate_demand!(M, input_share, Y)

`M[d,j,k] = input_share[d,j,k]·Y[d,k]` — country `d`'s demand for sector-`j` goods as an input
to its sector `k`.
"""
function _gvc_intermediate_demand!(M, input_share, Y)
    N, J = size(Y)
    @inbounds for k in 1:J, j in 1:J
        @views @. M[:, j, k] = input_share[:, j, k] * Y[:, k]
    end
    return M
end

"""
    _gvc_output!(Y, π_use, π_fin, M, F, τ, ζ)

Whitepaper equation (39) in levels:
`Y[o,j] = Σ_d ( π_fin[o,d,j]·F[d,j] + Σ_k π_use[o,d,j,k]·M[d,j,k] ) / (τ[o,d,j]·ζ[o,d,j])`.
"""
function _gvc_output!(Y, π_use, π_fin, M, F, τ, ζ)
    N, J = size(F)
    fill!(Y, 0.0)
    @inbounds for j in 1:J, d in 1:N
        f = F[d, j]
        if f != 0
            @views @. Y[:, j] += π_fin[:, d, j] * f / (τ[:, d, j] * ζ[:, d, j])
        end
        for k in 1:J
            mv = M[d, j, k]
            mv == 0 && continue
            @views @. Y[:, j] += π_use[:, d, j, k] * mv / (τ[:, d, j] * ζ[:, d, j])
        end
    end
    return Y
end

"""
    _gvc_aggregate_shares(π_use, π_fin, M, F, X) -> Array{Float64,3}

Expenditure-weighted average sourcing share,
`π̄[o,d,j] = (π_fin[o,d,j]·F[d,j] + Σ_k π_use[o,d,j,k]·M[d,j,k]) / X[d,j]`.

Because tariffs are product- rather than use-specific, this aggregate reproduces the same
bilateral flows and the same goods-market identity as the disaggregated shares, which is what
lets the ordinary [`KiteBaseline`](@ref) machinery apply unchanged.
"""
function _gvc_aggregate_shares(π_use, π_fin, M, F, X)
    N, _, J, _ = size(π_use)
    π̄ = zeros(N, N, J)
    @inbounds for j in 1:J, d in 1:N
        x = X[d, j]
        if x <= 0
            @views π̄[:, d, j] .= π_fin[:, d, j]
            continue
        end
        f = F[d, j]
        @views @. π̄[:, d, j] = π_fin[:, d, j] * f
        for k in 1:J
            mv = M[d, j, k]
            mv == 0 && continue
            @views @. π̄[:, d, j] += π_use[:, d, j, k] * mv
        end
        @views π̄[:, d, j] ./= x
    end
    return π̄
end

# ── workspace ─────────────────────────────────────────────────────────────────────────────

struct _ACWorkspace
    N::Int
    J::Int
    ŵ::Vector{Float64}
    ŵ_prev::Vector{Float64}
    I′::Vector{Float64}
    VA′::Vector{Float64}
    D′::Vector{Float64}
    ES′::Vector{Float64}
    factor_income::Vector{Float64}
    u::Vector{Float64}
    t::Vector{Float64}
    ĉ::Matrix{Float64}
    P̂_fin::Matrix{Float64}
    P̂_use::Array{Float64,3}
    P̂_use_prev::Array{Float64,3}         # (N, J, J) [d, j, k]: sector-j goods used by sector k
    logP̂_use::Array{Float64,3}
    X′::Matrix{Float64}
    Y′::Matrix{Float64}
    Y_prev::Matrix{Float64}
    F::Matrix{Float64}
    M::Array{Float64,3}
    tr_fin::Matrix{Float64}
    tr_use::Array{Float64,3}
    φ̂::Array{Float64,3}
    W_fin::Array{Float64,3}
    W_use::Array{Float64,4}
    A_fin::Array{Float64,3}
    A_use::Array{Float64,4}
    Aζ_fin::Array{Float64,3}
    Aζ_use::Array{Float64,4}
    has_export_subsidy::Bool
    has_tariff::Bool
    has_productivity::Bool
    has_population::Bool
end

function _ACWorkspace(g::GVCBaseline, sc::Scenario)
    b = g.base
    N, J = b.N, b.J

    φ̂ = similar(b.π)
    @inbounds @. φ̂ = (sc.τ′ / b.τ) * sc.κ̂ * (sc.ζ′ / b.ζ)

    W_fin = similar(b.π)
    @inbounds for j in 1:J
        θj = b.θ[j]
        @views @. W_fin[:, :, j] = g.π_fin[:, :, j] * φ̂[:, :, j]^(-θj)
    end
    W_use = Array{Float64,4}(undef, N, N, J, J)
    @inbounds for k in 1:J, j in 1:J
        θj = b.θ[j]
        @views @. W_use[:, :, j, k] = g.π_use[:, :, j, k] * φ̂[:, :, j]^(-θj)
    end

    has_es = any(!=(1.0), sc.ζ′)
    Aζ_fin = has_es ? similar(b.π) : Array{Float64,3}(undef, 0, 0, 0)
    Aζ_use = has_es ? Array{Float64,4}(undef, N, N, J, J) :
                      Array{Float64,4}(undef, 0, 0, 0, 0)

    return _ACWorkspace(N, J,
        ones(N), ones(N), copy(b.I), copy(b.VA), copy(b.D), zeros(N), copy(b.VA),
        zeros(N), zeros(N),
        ones(N, J), ones(N, J), ones(N, J, J), ones(N, J, J), zeros(N, J, J),
        copy(b.X), copy(b.Y), similar(b.Y), similar(b.X), Array{Float64,3}(undef, N, J, J),
        zeros(N, J), zeros(N, J, J),
        φ̂, W_fin, W_use, similar(b.π), Array{Float64,4}(undef, N, N, J, J),
        Aζ_fin, Aζ_use,
        has_es, any(!=(1.0), sc.τ′), any(!=(1.0), sc.ẑ), any(!=(1.0), sc.L̂))
end

# ── solver ────────────────────────────────────────────────────────────────────────────────

"""
    update_equilibrium(::AntrasChor2018, g::GVCBaseline, sc = Scenario(g.base); kwargs...)

Solve the Antràs–Chor global-value-chain equilibrium. Takes a [`GVCBaseline`](@ref); everything
else works as in the base [`update_equilibrium`](@ref).

The returned [`KiteResult`](@ref) reports aggregate quantities in its usual fields — `π′` is the
expenditure-weighted average sourcing share and `P̂` the final-consumption price index — so every
results helper applies unchanged. The use-specific detail is in `r.ext`: `π_use′`, `π_fin′` and
`P̂_use`.
"""
function update_equilibrium(model::AntrasChor2018, g::GVCBaseline,
                            sc::Scenario = Scenario(g.base); kwargs...)
    return update_equilibrium(model, g, sc, SolverSettings(; kwargs...))
end

function update_equilibrium(model::AntrasChor2018, g::GVCBaseline, sc::Scenario,
                            settings::SolverSettings)
    t0 = time()
    b = g.base
    _check_conformable(b, sc)
    settings.inner_solver === :iterative ||
        error("inner_solver must be :iterative for antras_chor_2018; the use-specific " *
              "expenditure block is not assembled as a dense linear system.")

    ws = _ACWorkspace(g, sc)
    num_index = _numeraire_index(b, settings.numeraire)

    criterion = Inf
    inner_ok = true
    inner_total = 0
    iter = 0

    while iter < settings.max_iterations
        iter += 1
        copyto!(ws.ŵ_prev, ws.ŵ)
        copyto!(ws.P̂_use_prev, ws.P̂_use)

        _ac_input_cost!(ws, b, sc)
        _ac_price_index!(ws, b)
        _ac_policy_weights!(ws, b, sc)
        @inbounds for d in 1:ws.N
            lw = ws.has_population ? sc.L̂[d] * ws.ŵ[d] : ws.ŵ[d]
            ws.factor_income[d] = lw * b.VA[d]
        end

        n_inner, inner_ok = _ac_inner!(ws, b, settings)
        inner_total += n_inner

        fill!(ws.VA′, 0.0)
        @inbounds for j in 1:ws.J
            @views @. ws.VA′ += b.β[:, j] * ws.Y′[:, j]
        end
        _ac_trade_balance!(ws, b, settings.trade_balance_rule)

        v = settings.vfactor
        @inbounds for o in 1:ws.N
            b.VA[o] > 0 || continue
            target = ws.VA′[o] / b.VA[o]
            ws.has_population && (target /= sc.L̂[o])
            target > 0 || continue
            ws.ŵ[o] = exp((1 - v) * log(ws.ŵ[o]) + v * log(target))
        end
        _ac_normalise!(ws, b, num_index)

        # see the note in solve.jl: wages alone can be at their fixed point while prices are
        # still propagating, so the price block enters the criterion too
        criterion = max(_criterion(ws.ŵ, ws.ŵ_prev, settings.convergence),
                        _criterion(ws.P̂_use, ws.P̂_use_prev, settings.convergence))
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
            @info @sprintf("antras_chor_2018: converged in %d iterations (criterion %.2e, \
                            %.3g s)", iter, criterion, elapsed)
        else
            @warn @sprintf("antras_chor_2018: did NOT converge in %d iterations \
                            (criterion %.2e).", iter, criterion)
        end
    end

    π_use′, π_fin′ = _ac_recover_shares(ws, sc)
    π̄ = _gvc_aggregate_shares(π_use′, π_fin′, ws.M, ws.F, ws.X′)

    return KiteResult(model, b, sc, settings,
                      copy(ws.ŵ), copy(ws.ĉ), copy(ws.P̂_fin), π̄,
                      copy(ws.X′), copy(ws.Y′), copy(ws.I′), copy(ws.VA′),
                      copy(ws.D′), zeros(ws.N),
                      converged, criterion, iter, inner_total, elapsed,
                      (π_use′ = π_use′, π_fin′ = π_fin′, P̂_use = copy(ws.P̂_use)))
end

"""
    _ac_input_cost!(ws, b, sc)

Whitepaper equation (34). Sector `j` is priced with the price indices *it* faces:
`ĉ[d,j] = exp( β[d,j]·log ŵ[d] + Σ_k input_share[d,k,j]·log P̂_use[d,k,j] )`, where
`P̂_use[d,k,j]` is the index for sector-`k` goods bought by sector `j`.
"""
function _ac_input_cost!(ws::_ACWorkspace, b::KiteBaseline, sc::Scenario)
    N, J = ws.N, ws.J
    @inbounds @. ws.logP̂_use = log(ws.P̂_use)
    @inbounds for j in 1:J
        cj = view(ws.ĉ, :, j)
        fill!(cj, 0.0)
        for k in 1:J
            @views @. cj += b.input_share[:, k, j] * ws.logP̂_use[:, k, j]
        end
        @views @. cj = exp(b.β[:, j] * log(ws.ŵ) + cj)
    end
    ws.has_productivity && @inbounds @. ws.ĉ /= sc.ẑ
    return ws
end

"""
    _ac_price_index!(ws, b)

Whitepaper equations (35) and (36): one price index per `(destination, supplying sector, using
sector)` plus one for final consumption, each a `gemv!` against the cached `W = π·φ̂^(-θ)`.
"""
function _ac_price_index!(ws::_ACWorkspace, b::KiteBaseline)
    N, J = ws.N, ws.J
    @inbounds for j in 1:J
        θj = b.θ[j]
        @views @. ws.u = ws.ĉ[:, j]^(-θj)
        @views mul!(ws.t, transpose(view(ws.W_fin, :, :, j)), ws.u)
        @views @. ws.P̂_fin[:, j] = ifelse(ws.t > 0, ws.t^(-1 / θj), 1.0)
        for k in 1:J
            @views mul!(ws.t, transpose(view(ws.W_use, :, :, j, k)), ws.u)
            @views @. ws.P̂_use[:, j, k] = ifelse(ws.t > 0, ws.t^(-1 / θj), 1.0)
        end
    end
    return ws
end

"""
    _ac_policy_weights!(ws, b, sc)

Build `A = π′/(τ′ζ′)` for both flows directly from the cached `W`, together with the
tariff-revenue weights `tr[d,j,·] = Σ_o (τ′−1)/τ′·π′` used by the inner loop.
"""
function _ac_policy_weights!(ws::_ACWorkspace, b::KiteBaseline, sc::Scenario)
    N, J = ws.N, ws.J
    @inbounds for j in 1:J
        θj = b.θ[j]
        @views @. ws.u = ws.ĉ[:, j]^(-θj)
        for d in 1:N
            vd = ws.P̂_fin[d, j]^θj
            @views @. ws.A_fin[:, d, j] =
                ws.W_fin[:, d, j] * ws.u * vd / (sc.τ′[:, d, j] * sc.ζ′[:, d, j])
            for k in 1:J
                vk = ws.P̂_use[d, j, k]^θj
                @views @. ws.A_use[:, d, j, k] =
                    ws.W_use[:, d, j, k] * ws.u * vk / (sc.τ′[:, d, j] * sc.ζ′[:, d, j])
            end
        end
    end

    if ws.has_export_subsidy
        @inbounds for j in 1:J, d in 1:N
            @views @. ws.Aζ_fin[:, d, j] = (sc.ζ′[:, d, j] - 1) * ws.A_fin[:, d, j]
            for k in 1:J
                @views @. ws.Aζ_use[:, d, j, k] = (sc.ζ′[:, d, j] - 1) * ws.A_use[:, d, j, k]
            end
        end
    end

    if ws.has_tariff
        @inbounds for j in 1:J, d in 1:N
            afin = 0.0
            @simd for o in 1:N
                afin += (sc.τ′[o, d, j] - 1) * sc.ζ′[o, d, j] * ws.A_fin[o, d, j]
            end
            ws.tr_fin[d, j] = afin
            for k in 1:J
                acc = 0.0
                @simd for o in 1:N
                    acc += (sc.τ′[o, d, j] - 1) * sc.ζ′[o, d, j] * ws.A_use[o, d, j, k]
                end
                ws.tr_use[d, j, k] = acc
            end
        end
    else
        fill!(ws.tr_fin, 0.0)
        fill!(ws.tr_use, 0.0)
    end
    return ws
end

"""
    _ac_inner!(ws, b, settings) -> (iterations, converged)

Warm-started fixed point in `(Y′, I′)`, whitepaper equations (39) and (41):

    F[d,j]   = α[d,j]·I′[d]
    M[d,j,k] = input_share[d,j,k]·Y′[d,k]
    Y′[o,j]  = Σ_d ( A_fin[o,d,j]·F[d,j] + Σ_k A_use[o,d,j,k]·M[d,j,k] )
    I′[d]    = Σ_j tr_fin[d,j]·F[d,j] + Σ_{j,k} tr_use[d,j,k]·M[d,j,k] + ES′[d]
             + factor_income[d] − D′[d]
"""
function _ac_inner!(ws::_ACWorkspace, b::KiteBaseline, settings::SolverSettings)
    N, J = ws.N, ws.J
    converged = false
    it = 0

    while it < settings.max_inner_iterations
        it += 1
        copyto!(ws.Y_prev, ws.Y′)

        @inbounds for j in 1:J
            @views @. ws.F[:, j] = b.α[:, j] * ws.I′
        end
        _gvc_intermediate_demand!(ws.M, b.input_share, ws.Y′)

        # gross output
        fill!(ws.Y′, 0.0)
        @inbounds for j in 1:J
            @views mul!(ws.t, view(ws.A_fin, :, :, j), ws.F[:, j])
            @views @. ws.Y′[:, j] += ws.t
            for k in 1:J
                @views mul!(ws.t, view(ws.A_use, :, :, j, k), ws.M[:, j, k])
                @views @. ws.Y′[:, j] += ws.t
            end
        end

        # export-subsidy cost accruing to the origin
        if ws.has_export_subsidy
            fill!(ws.ES′, 0.0)
            @inbounds for j in 1:J
                @views mul!(ws.t, view(ws.Aζ_fin, :, :, j), ws.F[:, j])
                @. ws.ES′ += ws.t
                for k in 1:J
                    @views mul!(ws.t, view(ws.Aζ_use, :, :, j, k), ws.M[:, j, k])
                    @. ws.ES′ += ws.t
                end
            end
        end

        # income
        @inbounds for d in 1:N
            acc = 0.0
            for j in 1:J
                acc += ws.tr_fin[d, j] * ws.F[d, j]
                @simd for k in 1:J
                    acc += ws.tr_use[d, j, k] * ws.M[d, j, k]
                end
            end
            ws.I′[d] = acc + (ws.has_export_subsidy ? ws.ES′[d] : 0.0) +
                       ws.factor_income[d] - ws.D′[d]
        end

        crit = _criterion(ws.Y′, ws.Y_prev, settings.convergence)
        isfinite(crit) || break
        if crit ≤ settings.inner_tolerance
            converged = true
            break
        end
    end

    # total expenditure by sector, for reporting
    @inbounds for j in 1:J
        @views @. ws.X′[:, j] = ws.F[:, j]
        for k in 1:J
            @views @. ws.X′[:, j] += ws.M[:, j, k]
        end
    end
    return it, converged
end

function _ac_trade_balance!(ws::_ACWorkspace, b::KiteBaseline, rule::Symbol)
    if rule === :fixed
        copyto!(ws.D′, b.D)
    elseif rule === :zero
        fill!(ws.D′, 0.0)
    elseif rule === :fixed_country_share
        @inbounds @. ws.D′ = ifelse(b.VA == 0, 0.0, b.D / b.VA * ws.VA′)
    elseif rule === :fixed_global_share
        @inbounds @. ws.D′ = b.D / sum(b.VA) * sum(ws.VA′)
    else
        error("trade_balance_rule must be :fixed, :fixed_country_share, :fixed_global_share " *
              "or :zero; got :$rule.")
    end
    return ws
end

function _ac_normalise!(ws::_ACWorkspace, b::KiteBaseline, num_index::Int)
    scale = if num_index == 0
        s = sum(ws.VA′)
        s > 0 ? sum(b.VA) / s : 1.0
    else
        w = ws.ŵ[num_index]
        w > 0 ? 1 / w : 1.0
    end
    isfinite(scale) && scale > 0 || return ws
    @inbounds @. ws.ŵ *= scale
    return ws
end

# π′ = A·(τ′ζ′), recovered once at the end rather than carried through the loop.
function _ac_recover_shares(ws::_ACWorkspace, sc::Scenario)
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
