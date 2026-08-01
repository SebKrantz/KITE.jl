# Kernels for use-specific sourcing, shared by the models that distinguish who is buying:
# [`AntrasChor2018`](@ref) and [`FelbermayrEtAl2025`](@ref).
#
# Both replace Caliendo & Parro's single trade share π[o,d,j] with one share per *using* sector,
# π[o,d,j,k], plus π[o,d,j,C] for final consumption. Tariffs stay product-specific — τ[o,d,j],
# not τ[o,d,j,k] — which matches how trade policy actually applies and is what lets the
# expenditure-weighted aggregate below reproduce the same bilateral flows and the same
# goods-market identity as the disaggregated shares.

"""
    _use_intermediate_demand!(M, input_share, Y)

`M[d,j,k] = input_share[d,j,k]·Y[d,k]` — country `d`'s demand for sector-`j` goods as an input
to its sector `k`.
"""
function _use_intermediate_demand!(M, input_share, Y)
    N, J = size(Y)
    @inbounds for k in 1:J, j in 1:J
        @views @. M[:, j, k] = input_share[:, j, k] * Y[:, k]
    end
    return M
end

"""
    _use_output!(Y, π_use, π_fin, M, F, τ, ζ)

Gross output from use-specific sourcing,

    Y[o,j] = Σ_d ( π_fin[o,d,j]·F[d,j] + Σ_k π_use[o,d,j,k]·M[d,j,k] ) / (τ[o,d,j]·ζ[o,d,j]),

where `F` is final demand and `M[d,j,k]` is intermediate demand for sector-`j` goods by sector
`k` in `d`.
"""
function _use_output!(Y, π_use, π_fin, M, F, τ, ζ)
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
    _use_aggregate_shares(π_use, π_fin, M, F, X) -> Array{Float64,3}

Expenditure-weighted average sourcing share,

    π̄[o,d,j] = ( π_fin[o,d,j]·F[d,j] + Σ_k π_use[o,d,j,k]·M[d,j,k] ) / X[d,j].

Because tariffs are product- rather than use-specific, this aggregate carries the same bilateral
flows and satisfies the same goods-market identity as the disaggregated shares — which is what
lets the ordinary [`KiteBaseline`](@ref) machinery apply to a use-specific economy unchanged.
"""
function _use_aggregate_shares(π_use, π_fin, M, F, X)
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

"""
    _normalise_use_shares!(pu, fallback = nothing)

Rescale each `(destination, supplying sector, using sector)` column of a `(N,N,J,J)` sourcing
array to sum to one over origins.

A column that is entirely zero is a market with no recorded trade. `fallback === nothing` closes
it to home supply, `pu[d,d,j,k] = 1`; otherwise the column is taken from `fallback[:,d,j]`, an
aggregate share array. Either way the choice is immaterial to the equilibrium — expenditure in
those cells is zero — and only keeps the price index well defined.
"""
function _normalise_use_shares!(pu, fallback = nothing)
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
        elseif fallback === nothing
            pu[d, d, j, k] = 1.0
        else
            @simd for o in 1:N
                pu[o, d, j, k] = fallback[o, d, j]
            end
        end
    end
    return pu
end
