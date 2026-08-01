# Chowdhry, Hinz, Kamin & Wanner (2022/2024).
#
# Adds transfers among a sanctioning coalition S, pinned down by two conditions: the transfers
# net to zero within the coalition, and every member ends up with the same proportional welfare
# change. The wage iterate is updated from the trade-balance excess-demand function instead of
# labour-market clearing.
#
# Three corrections relative to the R implementation (see NEWS.md):
#   * exports and imports are deflated by τ′·ζ′, not by τ′ alone;
#   * the transfer step uses the counterfactual trade balance D′, not the baseline D;
#   * the wage vector is renormalised against the numéraire each iteration, as in CP2015.

function _check_scenario(::ChowdhryHinzKaminWanner2022, b::KiteBaseline, sc::Scenario)
    if !any(sc.coalition)
        @info "chowdhry_hinz_kamin_wanner_2022: the coalition is empty, so no transfers are \
               made and this solves the same equilibrium as caliendo_parro_2015. Use \
               `set_coalition!` to add members."
    end
    return nothing
end

"""
    _transfers!(::ChowdhryHinzKaminWanner2022, ws, b, sc)

Transfers `T′` across the coalition `S`, satisfying `Σ_{d∈S} T′_d = 0` and equal welfare change
`Î_d/P̂_d = c̄` for all `d ∈ S`:

    P̂_d = exp( Σ_j α[d,j]·log P̂[d,j] )
    c̄   = Σ_{d∈S} I′_d / Σ_{d∈S} (I_d · P̂_d)
    T′_d = c̄·(I_d · P̂_d) − I′_d          for d ∈ S,  0 otherwise

`I′` here excludes the transfer itself, so it is recomputed net of the previous iterate.
"""
function _transfers!(::ChowdhryHinzKaminWanner2022, ws::_Workspace, b::KiteBaseline,
                     sc::Scenario)
    any(sc.coalition) || return ws
    N, J = ws.N, ws.J

    @inbounds for d in 1:N
        acc = 0.0
        @simd for j in 1:J
            acc += b.α[d, j] * log(ws.P̂[d, j])
        end
        ws.P̂_country[d] = exp(acc)
    end

    num = 0.0
    den = 0.0
    @inbounds for d in 1:N
        sc.coalition[d] || continue
        num += ws.I′[d] - ws.T′[d]          # income excluding the current transfer
        den += b.I[d] * ws.P̂_country[d]
    end
    den > 0 || return ws
    c̄ = num / den

    @inbounds for d in 1:N
        ws.T′[d] = sc.coalition[d] ?
            c̄ * b.I[d] * ws.P̂_country[d] - (ws.I′[d] - ws.T′[d]) : 0.0
    end
    return ws
end

"""
    _wage_update!(::ChowdhryHinzKaminWanner2022, ws, b, sc, settings)

Excess demand on the trade balance, and an additive step in its direction:

    exports[o] = Σ_{d,j} π′[o,d,j]·X′[d,j] / (τ′[o,d,j]·ζ′[o,d,j])  ( = Σ_j Y′[o,j] )
    imports[d] = Σ_{o,j} π′[o,d,j]·X′[d,j] / (τ′[o,d,j]·ζ′[o,d,j])
    excess[d]  = (exports[d] − imports[d] − D′[d] + T′[d]) / VA[d]
    ŵ[d]       ← max(ŵ[d] + v·excess[d], eps())
"""
function _wage_update!(::ChowdhryHinzKaminWanner2022, ws::_Workspace, b::KiteBaseline,
                       sc::Scenario, settings::SolverSettings)
    N, J = ws.N, ws.J

    # exports: Y′ is already Σ_d A[o,d,j]·X′[d,j], so summing over sectors gives fob exports
    fill!(ws.exports, 0.0)
    @inbounds for j in 1:J
        @views @. ws.exports += ws.Y′[:, j]
    end

    # imports: for each destination, sum the same fob flows over origins
    fill!(ws.imports, 0.0)
    @inbounds for j in 1:J, d in 1:N
        acc = 0.0
        x = ws.X′[d, j]
        @simd for o in 1:N
            acc += ws.A[o, d, j] * x
        end
        ws.imports[d] += acc
    end

    v = settings.vfactor
    @inbounds for d in 1:N
        va = b.VA[d]
        va > 0 || continue
        ws.excess[d] = (ws.exports[d] - ws.imports[d] - ws.D′[d] + ws.T′[d]) / va
        ws.ŵ[d] = max(ws.ŵ[d] + v * ws.excess[d], eps())
    end
    return ws
end
