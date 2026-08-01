# Caliendo & Parro (2015).
#
# The wage iterate follows labour-market clearing: value added implied by the new output must
# equal the wage bill. The update is taken in logs and dampened by `vfactor`, then the
# numéraire is imposed by the shared outer loop.

_check_scenario(::CaliendoParro2015, b::KiteBaseline, sc::Scenario) = nothing

# No transfers in CP2015; T′ stays at zero.
_transfers!(::CaliendoParro2015, ws::_Workspace, b::KiteBaseline, sc::Scenario) = ws

"""
    _wage_update!(::CaliendoParro2015, ws, b, sc, settings)

`ŵ_o ← exp( (1 − v)·log ŵ_o + v·log( (VA′_o / VA_o) / L̂_o ) )`

The geometric form keeps `ŵ` positive for any dampening factor `v = vfactor`.
"""
function _wage_update!(::CaliendoParro2015, ws::_Workspace, b::KiteBaseline, sc::Scenario,
                       settings::SolverSettings)
    v = settings.vfactor
    @inbounds for o in 1:ws.N
        va = b.VA[o]
        va > 0 || continue
        target = ws.VA′[o] / va
        ws.has_population && (target /= sc.L̂[o])
        target > 0 || continue
        ws.ŵ[o] = exp((1 - v) * log(ws.ŵ[o]) + v * log(target))
    end
    return ws
end
