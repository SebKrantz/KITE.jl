# Solver scratch space.
#
# Everything the outer and inner loops touch is allocated once, here, and written in place
# afterwards. A steady-state outer iteration allocates nothing.
#
# Three quantities the R implementation rebuilds inside its loops are hoisted:
#
#   W[o,d,j]      = π[o,d,j] · φ̂[o,d,j]^(-θ_j)     — loop-invariant, built once per solve
#   A[o,d,j]      = π′[o,d,j] / (τ′[o,d,j] ζ′[o,d,j])  — once per *outer* iteration
#   tr_share[d,j] = Σ_o (τ′[o,d,j] − 1)/τ′[o,d,j] · π′[o,d,j]  — once per *outer* iteration
#
# `tr_share` matters most: tariff revenue is linear in X′ with π′-fixed weights, so collapsing
# it to an N×J matrix turns the inner loop's income step from O(N²J) into O(NJ). The R code
# builds three full (N,N,J) temporaries there on every inner iteration.

struct _Workspace{E}
    N::Int
    J::Int
    # country vectors
    ŵ::Vector{Float64}
    ŵ_prev::Vector{Float64}
    I′::Vector{Float64}
    VA′::Vector{Float64}
    D′::Vector{Float64}
    T′::Vector{Float64}
    ES′::Vector{Float64}
    excess::Vector{Float64}
    P̂_country::Vector{Float64}
    exports::Vector{Float64}
    imports::Vector{Float64}
    u::Vector{Float64}
    t::Vector{Float64}
    # payments to primary factors, ŵ·VA in the baseline models and ŵ·L + Σ_p p̂·R under
    # Mahlkow & Wanner. Fixed during the inner loop, so it is computed once per outer pass.
    factor_income::Vector{Float64}
    # country × sector matrices
    ĉ::Matrix{Float64}
    P̂_prev::Matrix{Float64}
    P̂::Matrix{Float64}
    logP̂::Matrix{Float64}
    X′::Matrix{Float64}
    Y′::Matrix{Float64}
    Y_prev::Matrix{Float64}
    ID::Matrix{Float64}
    tr_share::Matrix{Float64}
    # bilateral tensors
    φ̂::Array{Float64,3}
    W::Array{Float64,3}
    π′::Array{Float64,3}
    A::Array{Float64,3}
    Aζ::Array{Float64,3}
    # fast-path flags, computed once
    has_export_subsidy::Bool
    has_tariff::Bool
    has_productivity::Bool
    has_population::Bool
    # model-specific scratch; `nothing` for models that need none
    ext::E
end

"""
    _refresh_wedges!(φ̂, W, b, sc)

Rebuild the two loop-invariants that depend on the policy wedges,
`φ̂ = (τ′/τ)·κ̂·(ζ′/ζ)` and `W = π·φ̂^(-θ)`.

Normally this runs once per solve. It has to run once per *outer iteration* when a specific
carbon price is in force, because that wedge is `1 + price·χ/P̂` and so moves with the fuel
price. The cost is one `O(N²J)` pass, the same order as the price-index step it feeds.
"""
function _refresh_wedges!(φ̂::AbstractArray{Float64,3}, W::AbstractArray{Float64,3},
                          b::KiteBaseline, sc::Scenario)
    @inbounds @. φ̂ = (sc.τ′ / b.τ) * sc.κ̂ * (sc.ζ′ / b.ζ)
    @inbounds for j in axes(W, 3)
        θj = b.θ[j]
        @views @. W[:, :, j] = b.π[:, :, j] * φ̂[:, :, j]^(-θj)
    end
    return W
end

function _Workspace(b::KiteBaseline, sc::Scenario, ext = nothing)
    N, J = b.N, b.J

    φ̂ = similar(b.π)
    W = similar(b.π)
    _refresh_wedges!(φ̂, W, b, sc)

    has_export_subsidy = any(!=(1.0), sc.ζ′)
    Aζ = has_export_subsidy ? similar(b.π) : Array{Float64,3}(undef, 0, 0, 0)

    # Warm start every level at the calibrated baseline, including income. Starting I′ at zero
    # would cost the inner loop its first few passes even when nothing has changed, and would
    # leave the no-change scenario short of machine precision.
    return _Workspace(N, J,
        ones(N), ones(N), copy(b.I), copy(b.VA), copy(b.D), zeros(N), zeros(N), zeros(N),
        ones(N), zeros(N), zeros(N), zeros(N), zeros(N), copy(b.VA),
        ones(N, J), ones(N, J), ones(N, J), zeros(N, J),
        copy(b.X), copy(b.Y), similar(b.Y), zeros(N, J), zeros(N, J),
        φ̂, W, copy(b.π), similar(b.π), Aζ,
        has_export_subsidy,
        any(!=(1.0), sc.τ′),
        any(!=(1.0), sc.ẑ),
        any(!=(1.0), sc.L̂),
        ext)
end
