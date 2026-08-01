# Calibration: turn raw MRIO data into a model-consistent baseline.
#
# The R implementation takes the supplied levels at face value. On the shipped 2022 database
# those levels do not satisfy the model's own accounting identities — `consumption_share` is
# built from household final consumption only, while `expenditure` also contains government,
# investment and inventory demand — so a no-change scenario does not reproduce the baseline.
# `calibrate` closes that gap before the solver ever runs.

"""
    residuals(b::KiteBaseline) -> NamedTuple

Maximum relative violation of the three equilibrium identities at the baseline:

* `goods_market` — `Y[o, j] = Σ_d π[o, d, j] · X[d, j] / (τ[o, d, j] · ζ[o, d, j])`
* `expenditure`  — `X[d, k] = α[d, k] · I[d] + Σ_j input_share[d, k, j] · Y[d, j]`
* `income`       — `I[d] = VA[d] + R[d] − D[d]`

A baseline built by [`calibrate`](@ref) returns values at round-off. The
[`KiteBaseline`](@ref) constructor refuses to build anything larger than its `atol`.
"""
function residuals(b::KiteBaseline)
    Y = _gross_output(b.π, b.X, b.τ, b.ζ)
    goods = _max_rel_dev(Y, b.Y)

    X = similar(b.X)
    _intermediate_demand!(X, b.input_share, b.Y)
    @inbounds @. X += b.α * b.I
    expend = _max_rel_dev(X, b.X)

    inc = 0.0
    @inbounds for d in 1:b.N
        va = 0.0
        for j in 1:b.J
            va += b.β[d, j] * b.Y[d, j]
        end
        target = va + b.R[d] - b.D[d]
        inc = max(inc, abs(target - b.I[d]) / max(abs(b.I[d]), eps()))
    end

    return (goods_market = goods, expenditure = expend, income = inc)
end

function _max_rel_dev(a, b)
    m = 0.0
    @inbounds for i in eachindex(a, b)
        d = abs(a[i] - b[i]) / max(abs(b[i]), eps())
        m = ifelse(d > m, d, m)
    end
    return m
end

"""
    _gross_output(π, X, τ, ζ) -> Matrix

`Y[o, j] = Σ_d π[o, d, j] · X[d, j] / (τ[o, d, j] · ζ[o, d, j])`, the free-on-board value of
each origin's sales.
"""
function _gross_output(π, X, τ, ζ)
    N, _, J = size(π)
    Y = zeros(N, J)
    @inbounds for j in 1:J, d in 1:N
        x = X[d, j]
        x == 0 && continue
        @views @. Y[:, j] += π[:, d, j] * x / (τ[:, d, j] * ζ[:, d, j])
    end
    return Y
end

"""
    _intermediate_demand!(ID, input_share, Y)

`ID[d, k] = Σ_j input_share[d, k, j] · Y[d, j]`, demand for sector-`k` goods as intermediates.
Both operands are indexed country-first, so the inner slice is contiguous.
"""
function _intermediate_demand!(ID, input_share, Y)
    N, J = size(Y)
    fill!(ID, 0.0)
    @inbounds for j in 1:J, k in 1:J
        @views @. ID[:, k] += input_share[:, k, j] * Y[:, j]
    end
    return ID
end

"""
    _revenue(π, X, τ, ζ) -> Vector

Net policy revenue `R[d]`: tariffs collected by `d` on its imports, plus the (negative) cost of
export subsidies `d` pays on its exports.

    R[d] = Σ_{o,j} (τ[o,d,j] − 1)/τ[o,d,j] · π[o,d,j] · X[d,j]
         + Σ_{m,j} (ζ[d,m,j] − 1)/(τ[d,m,j]·ζ[d,m,j]) · π[d,m,j] · X[m,j]
"""
function _revenue(π, X, τ, ζ)
    N, _, J = size(π)
    R = zeros(N)
    @inbounds for j in 1:J
        for d in 1:N
            x = X[d, j]
            x == 0 && continue
            acc = 0.0
            @simd for o in 1:N
                t = τ[o, d, j]
                acc += (t - 1) / t * π[o, d, j]
            end
            R[d] += acc * x
        end
        for m in 1:N
            x = X[m, j]
            x == 0 && continue
            @simd for d in 1:N
                z = ζ[d, m, j]
                R[d] += (z - 1) / (τ[d, m, j] * z) * π[d, m, j] * x
            end
        end
    end
    return R
end

"""
    calibrate(; countries, sectors, π, γ, β, θ, X, α = nothing, τ = nothing, ζ = nothing,
                anchor = :expenditure, final_demand = :residual, kwargs...) -> KiteBaseline

Build a model-consistent [`KiteBaseline`](@ref) from raw shares and levels.

The equilibrium identities over-determine raw MRIO data, so something has to give. `calibrate`
holds the behavioural shares (`π`, `γ`, `β`, `θ`) and the observed expenditure `X` fixed, and
lets final demand and the trade balance absorb the inconsistency:

1. `Y[o, j] = Σ_d π · X / (τ ζ)` and `ID[d, k] = Σ_j input_share · Y`.
2. `F = X − ID` is residual final demand. Where a cell is negative (inventory decumulation and
   MRIO adjustment items), it is clipped to zero, `X` is reset to `ID + F⁺`, and the step is
   repeated until `F ≥ 0` everywhere. This converges geometrically.
3. `α = F / Σ_j F` — final-absorption shares that reproduce observed final demand by
   construction.
4. `VA = Σ_j β · Y`, `R` from the policy wedges, `I = Σ_j F`, and the trade balance closes the
   income identity as `D = VA + R − I`.

# Keyword Arguments
- `countries`, `sectors`: label vectors of length `N` and `J`.
- `π`: `(N, N, J)` `[o, d, j]` trade shares. Columns are renormalised to sum to one.
- `γ`: `(N, J, J)` `[d, k, j]` intermediate shares, `k` input and `j` output. Renormalised to
  sum to one over `k` (a column of zeros, for an inactive sector, is left alone).
- `β`: `(N, J)` value-added shares, clipped into `(clip_β, 1 − clip_β)`.
- `θ`: `(J)` trade elasticities in the **standard** convention — larger `θ` means more
  responsive trade. (The R implementation inverts this; see `NEWS.md`.)
- `X`: `(N, J)` expenditure. Required when `anchor = :expenditure`.
- `α`: `(N, J)` final-absorption shares. Only used when `final_demand = :given`.
- `τ`, `ζ`: `(N, N, J)` tariff and export tax/subsidy multipliers; default to no policy.
- `anchor::Symbol = :expenditure`: `:expenditure` holds `X` fixed (recommended, and what the
  shipped baseline uses); `:value_added` instead holds published `VA` fixed and solves jointly
  for `X`, letting expenditure move.
- `VA`, `D`: `(N)` published value added and trade surplus. Used when `anchor = :value_added`;
  ignored otherwise (both are then implied).
- `final_demand::Symbol = :residual`: `:residual` derives `α` as above; `:given` uses the
  supplied `α` and reports how far the resulting baseline sits from the data.
- `clip_negative::Bool = true`: clip negative trade shares (supply-use balancing items) to zero
  before renormalising, reporting how many and how much.
- `clip_β::Float64 = 1e-4`: bound keeping `β` strictly inside `(0, 1)`.
- `tolerance::Float64 = 1e-12`, `max_iterations::Int = 1000`: balancing-loop controls.
- `verbose::Int = 1`: `1` prints a one-line calibration summary.

# Examples
```julia
b = calibrate(; countries, sectors, π, γ, β, θ, X)
residuals(b)   # (goods_market = 2.6e-16, expenditure = 0.0, income = 0.0)
```
"""
function calibrate(; countries, sectors,
                     π, γ, β, θ,
                     X = nothing, α = nothing, VA = nothing, D = nothing,
                     τ = nothing, ζ = nothing,
                     anchor::Symbol = :expenditure,
                     final_demand::Symbol = :residual,
                     clip_negative::Bool = true,
                     clip_β::Float64 = 1e-4,
                     tolerance::Float64 = 1e-12,
                     max_iterations::Int = 1000,
                     verbose::Int = 1)

    anchor in (:expenditure, :value_added) ||
        error("anchor must be :expenditure or :value_added; got :$anchor.")
    final_demand in (:residual, :given) ||
        error("final_demand must be :residual or :given; got :$final_demand.")

    N = length(countries)
    J = length(sectors)
    _check_size(π, (N, N, J), "π")
    _check_size(γ, (N, J, J), "γ")
    _check_size(β, (N, J), "β")
    _check_size(θ, (J,), "θ")

    π = Array{Float64,3}(π)
    γ = Array{Float64,3}(γ)
    β = clamp.(Float64.(β), clip_β, 1 - clip_β)
    θ = Vector{Float64}(θ)
    τ = τ === nothing ? ones(N, N, J) : Array{Float64,3}(τ)
    ζ = ζ === nothing ? ones(N, N, J) : Array{Float64,3}(ζ)
    _check_size(τ, (N, N, J), "τ")
    _check_size(ζ, (N, N, J), "ζ")

    _normalise_trade_shares!(π; verbose = verbose, clip_negative = clip_negative)
    _normalise_intermediate_shares!(γ)

    input_share = similar(γ)
    @inbounds for j in 1:J, k in 1:J
        @views @. input_share[:, k, j] = (1 - β[:, j]) * γ[:, k, j]
    end

    if anchor === :value_added
        VA === nothing && error("anchor = :value_added requires VA.")
        _check_size(VA, (N,), "VA")
        X = _expenditure_from_value_added(π, input_share, β, τ, ζ, Float64.(VA),
                                          D === nothing ? zeros(N) : Float64.(D),
                                          α, final_demand, N, J, tolerance, max_iterations)
    else
        X === nothing && error("anchor = :expenditure requires X.")
        _check_size(X, (N, J), "X")
        X = max.(Float64.(X), 0.0)
    end

    # ── balancing loop: drive residual final demand non-negative ──────────────────────────
    # Iterate until residual final demand is non-negative everywhere. The loop exits only when
    # `F = X - ID ≥ 0` holds for the *current* X, so X, Y, ID and F are mutually consistent on
    # exit and the goods-market identity is exact — no post-hoc clipping, which would otherwise
    # move X away from the Y it was derived from.
    ID = zeros(N, J)
    Y = zeros(N, J)
    F = zeros(N, J)
    passes = 0
    converged_balance = false
    for it in 1:max_iterations
        passes = it
        Y = _gross_output(π, X, τ, ζ)
        _intermediate_demand!(ID, input_share, Y)
        @. F = X - ID
        if minimum(F) ≥ 0
            converged_balance = true
            break
        end
        @. X = ID + max(F, 0.0)
    end
    if !converged_balance
        @. F = max(F, 0.0)
        @. X = ID + F
        Y = _gross_output(π, X, τ, ζ)
        _intermediate_demand!(ID, input_share, Y)
        @. F = X - ID
        verbose ≥ 1 && @warn @sprintf(
            "calibrate: the balancing loop did not drive residual final demand non-negative \
             within %d passes (minimum %.3g). Raise max_iterations.", max_iterations,
            minimum(F))
    end

    # ── shares, levels, and the closing identity ──────────────────────────────────────────
    I = vec(sum(F, dims = 2))
    if final_demand === :given
        α === nothing && error("final_demand = :given requires α.")
        _check_size(α, (N, J), "α")
        α = Float64.(α)
        _normalise_rows!(α)
        dev = _max_rel_dev(α .* I, F)
        verbose ≥ 1 && @info @sprintf(
            "calibrate: using supplied α; implied final demand deviates from the data by up \
             to %.1f%% per cell. Consider final_demand = :residual.", 100 * dev)
        @. X = ID + α * I
        Y = _gross_output(π, X, τ, ζ)
        _intermediate_demand!(ID, input_share, Y)
        I = vec(sum(X - ID, dims = 2))
    else
        α = similar(F)
        @inbounds for d in 1:N
            s = I[d]
            if s > 0
                @views @. α[d, :] = F[d, :] / s
            else
                fill!(view(α, d, :), 1.0 / J)
            end
        end
    end

    R = _revenue(π, X, τ, ζ)
    VA_out = vec(sum(β .* Y, dims = 2))
    D_out = VA_out .+ R .- I

    b = KiteBaseline(countries, sectors, π, γ, α, β, θ, τ, ζ, X, Y, I, R, VA_out, D_out;
                     atol = 1e-8)

    if verbose ≥ 1
        res = residuals(b)
        @info @sprintf(
            "calibrate: %d balancing pass%s, residuals (goods %.1e, expenditure %.1e, \
             income %.1e), Σ D = %.3g on a base of %.3g",
            passes, passes == 1 ? "" : "es", res.goods_market, res.expenditure, res.income,
            sum(D_out), sum(VA_out))
    end
    return b
end

"""
    _normalise_trade_shares!(π; verbose = 1) -> π

Rescale each `(destination, sector)` column to sum to one.

A column that is entirely zero is a market with no recorded trade — in the OECD ICIO this is
sector `T` (activities of households), which has no gross output anywhere. Such a market is
degenerate rather than erroneous, so it is closed by convention: all demand is met from home,
`π[d, d, j] = 1`. Expenditure in those cells is zero, so the choice has no effect on the
equilibrium; it only keeps the price index well defined.
"""
function _normalise_trade_shares!(π; verbose::Int = 1, clip_negative::Bool = true)
    N, _, J = size(π)

    # Supply-use tables carry occasional negative entries (inventory decumulation and other
    # balancing items). Expenditure shares must be non-negative, so clip and renormalise. On
    # the shipped 2022 database this touches 8 of 301,871 cells, none larger than 0.036.
    if clip_negative
        n_neg = 0
        mass = 0.0
        @inbounds for i in eachindex(π)
            if π[i] < 0
                n_neg += 1
                mass -= π[i]
                π[i] = 0.0
            end
        end
        if n_neg > 0 && verbose ≥ 1
            @info @sprintf("calibrate: clipped %d negative trade shares to zero (total \
                            magnitude %.4g) before renormalising.", n_neg, mass)
        end
    end

    empty_markets = 0
    @inbounds for j in 1:J, d in 1:N
        s = 0.0
        @simd for o in 1:N
            s += π[o, d, j]
        end
        if s > 0
            @simd for o in 1:N
                π[o, d, j] /= s
            end
        else
            empty_markets += 1
            π[d, d, j] = 1.0
        end
    end
    if empty_markets > 0 && verbose ≥ 1
        @info "calibrate: $empty_markets (destination, sector) markets have no recorded " *
              "trade and were closed to home supply (π[d, d, j] = 1)."
    end
    return π
end

function _normalise_intermediate_shares!(γ)
    N, J, _ = size(γ)
    @inbounds for j in 1:J, d in 1:N
        s = 0.0
        @simd for k in 1:J
            s += γ[d, k, j]
        end
        s > 0 || continue   # inactive sector: leave the all-zero column alone
        @simd for k in 1:J
            γ[d, k, j] /= s
        end
    end
    return γ
end

function _normalise_rows!(A)
    @inbounds for d in axes(A, 1)
        s = sum(view(A, d, :))
        s > 0 && (view(A, d, :) ./= s)
    end
    return A
end

# anchor = :value_added — hold published VA (and D) fixed and solve the joint linear fixed
# point for X. Requires α, so it is only meaningful with final_demand = :given; with
# :residual we bootstrap from an equal-shares guess and iterate α along with X.
function _expenditure_from_value_added(π, input_share, β, τ, ζ, VA, D, α, final_demand,
                                       N, J, tolerance, max_iterations)
    α_use = if final_demand === :given && α !== nothing
        A = Float64.(α); _normalise_rows!(A); A
    else
        fill(1.0 / J, N, J)
    end
    X = repeat(VA ./ J, 1, J)
    ID = zeros(N, J)
    for _ in 1:max_iterations
        Y = _gross_output(π, X, τ, ζ)
        _intermediate_demand!(ID, input_share, Y)
        R = _revenue(π, X, τ, ζ)
        I = VA .+ R .- D
        X_new = ID .+ α_use .* I
        if _max_rel_dev(X_new, X) < tolerance
            X = X_new
            break
        end
        X = X_new
    end
    return X
end
