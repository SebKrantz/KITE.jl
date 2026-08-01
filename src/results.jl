# Post-processing.
#
# Everything is computed from the exact calibrated baseline levels stored in the KiteBaseline,
# not re-derived. (The R implementation re-runs its expenditure fixed point inside
# `process_results`, which both costs time and lets the reported baseline drift away from the
# one the solver actually used.)
#
# "fob" denotes the value received by the producer, i.e. deflated by τ·ζ — consistent with
# gross output Y. The R code deflates by τ alone, which differs whenever export taxes or
# subsidies are active.

"""
    price_index_change(r::KiteResult) -> Vector{Float64}

Change in the aggregate consumer price index, `P̂_d = Π_j (P̂_d^j)^{α_d^j}`.
"""
function price_index_change(r::KiteResult)
    b = r.baseline
    out = Vector{Float64}(undef, b.N)
    @inbounds for d in 1:b.N
        acc = 0.0
        @simd for j in 1:b.J
            acc += b.α[d, j] * log(r.P̂[d, j])
        end
        out[d] = exp(acc)
    end
    return out
end

"""
    price_index(r::KiteResult; weights = nothing) -> Vector{Float64}

Aggregate price-index change under arbitrary expenditure weights, `Π_j (P̂_d^j)^{w_d^j}`.

`weights` defaults to the baseline final-absorption shares `α`, reproducing
[`price_index_change`](@ref). Pass an `(N, J)` matrix — household consumption shares, say — to
report a price index over a different bundle. Rows are renormalised to sum to one.
"""
function price_index(r::KiteResult; weights = nothing)
    b = r.baseline
    weights === nothing && return price_index_change(r)
    _check_size(weights, (b.N, b.J), "weights")
    w = Matrix{Float64}(weights)
    _normalise_rows!(w)
    out = Vector{Float64}(undef, b.N)
    @inbounds for d in 1:b.N
        acc = 0.0
        @simd for j in 1:b.J
            acc += w[d, j] * log(r.P̂[d, j])
        end
        out[d] = exp(acc)
    end
    return out
end

"""
    welfare_change(r::KiteResult) -> Vector{Float64}

Change in real income, `Î_d / P̂_d`. Values above one are welfare gains.
"""
welfare_change(r::KiteResult) = (r.I′ ./ r.baseline.I) ./ price_index_change(r)

"""
    real_wage_change(r::KiteResult) -> Vector{Float64}

Change in the real wage, `ŵ_d / P̂_d`.
"""
real_wage_change(r::KiteResult) = r.ŵ ./ price_index_change(r)

"""
    tariff_revenue(r::KiteResult) -> (baseline, counterfactual)

Tariff revenue collected by each destination, `Σ_{o,j} (τ−1)/τ · π[o,d,j] · X[d,j]`.
"""
function tariff_revenue(r::KiteResult)
    b = r.baseline
    return _tariff_revenue(b.π, b.X, b.τ), _tariff_revenue(r.π′, r.X′, r.scenario.τ′)
end

function _tariff_revenue(π, X, τ)
    N, _, J = size(π)
    out = zeros(N)
    @inbounds for j in 1:J, d in 1:N
        x = X[d, j]
        x == 0 && continue
        acc = 0.0
        @simd for o in 1:N
            t = τ[o, d, j]
            acc += (t - 1) / t * π[o, d, j]
        end
        out[d] += acc * x
    end
    return out
end

"""
    export_subsidy_costs(r::KiteResult) -> (baseline, counterfactual)

Net export tax receipts (positive) or subsidy costs (negative) borne by each origin,
`Σ_{m,j} (ζ−1)/(τ ζ) · π[d,m,j] · X[m,j]`.
"""
function export_subsidy_costs(r::KiteResult)
    b = r.baseline
    return _subsidy_costs(b.π, b.X, b.τ, b.ζ), _subsidy_costs(r.π′, r.X′, r.scenario.τ′, r.scenario.ζ′)
end

function _subsidy_costs(π, X, τ, ζ)
    N, _, J = size(π)
    out = zeros(N)
    @inbounds for j in 1:J, m in 1:N
        x = X[m, j]
        x == 0 && continue
        @simd for d in 1:N
            z = ζ[d, m, j]
            out[d] += (z - 1) / (τ[d, m, j] * z) * π[d, m, j] * x
        end
    end
    return out
end

"""
    trade_flows(r::KiteResult) -> NamedTuple

Bilateral trade at baseline and counterfactual, gross (tariff-inclusive expenditure) and fob
(producer revenue, deflated by `τ·ζ`).

Returns `(flow, flow_new, fob, fob_new)`, each `(N, N, J)` indexed `[o, d, j]`.
"""
function trade_flows(r::KiteResult)
    b = r.baseline
    flow = _flow(b.π, b.X)
    flow_new = _flow(r.π′, r.X′)
    fob = _fob(flow, b.τ, b.ζ)
    fob_new = _fob(flow_new, r.scenario.τ′, r.scenario.ζ′)
    return (flow = flow, flow_new = flow_new, fob = fob, fob_new = fob_new)
end

function _flow(π, X)
    N, _, J = size(π)
    out = similar(π)
    @inbounds for j in 1:J, d in 1:N
        @views @. out[:, d, j] = π[:, d, j] * X[d, j]
    end
    return out
end

_fob(flow, τ, ζ) = @inbounds flow ./ (τ .* ζ)

"""
    trade_aggregates(r::KiteResult) -> NamedTuple

Country-level fob exports and imports, `(exports, exports_new, imports, imports_new)`.
"""
function trade_aggregates(r::KiteResult)
    b = r.baseline
    tf = trade_flows(r)
    return (exports = vec(sum(tf.fob, dims = (2, 3))),
            exports_new = vec(sum(tf.fob_new, dims = (2, 3))),
            imports = vec(sum(tf.fob, dims = (1, 3))),
            imports_new = vec(sum(tf.fob_new, dims = (1, 3))))
end

_safe_ratio(a, b) = ifelse(b == 0, ifelse(a == 0, 1.0, NaN), a / b)
