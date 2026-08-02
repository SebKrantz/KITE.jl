# 04c_stability.jl — why the 196x133 tariff counterfactual fails to converge.
#
# The countries with implausible results under the default settings (QAT, TKM, GNQ,
# BRN, COG, LBY, MNG, GIN, SUR) are the economies specialised in exactly the sectors
# carrying the largest elasticities: GAS (theta 64), COAL (33), HS26 ores (33). This
# script separates the two candidate explanations — a stiff elasticity tail versus
# plain under-damping — by varying one at a time.

using KITE, Printf, Statistics, Serialization

b0 = deserialize(joinpath(@__DIR__, "build", "2023", "baseline.jls"))
step(s) = (println("\n", "-"^72, "\n", s, "\n", "-"^72); flush(stdout))

"Rebuild the baseline with theta winsorised at `cap`."
function recap(b, cap)
    θ = min.(b.θ, cap)
    calibrate(; countries = b.countries, sectors = b.sectors, π = b.π, γ = b.γ,
                β = b.β, θ = θ, X = b.X, τ = b.τ, ζ = b.ζ, verbose = 0)
end

function trial(name, b, shock; kwargs...)
    sc = Scenario(b; label = name)
    set_tariff!(sc, b, shock; from = "CHN", to = "USA", mode = :add)
    t = @elapsed r = update_equilibrium(CaliendoParro2015(), b, sc; verbose = 0, kwargs...)
    @printf("%-34s  iters %5d  criterion %.3e  converged %-5s  %6.1f s\n",
            name, r.iterations, r.criterion, r.converged, t)
    flush(stdout)
    return r
end

@printf("theta: median %.2f  p90 %.2f  p99 %.2f  max %.2f\n",
        median(b0.θ), quantile(b0.θ, .9), quantile(b0.θ, .99), maximum(b0.θ))
@printf("sectors with theta > 20: %d -> %s\n", sum(b0.θ .> 20),
        join(b0.sectors[b0.θ .> 20], " "))
flush(stdout)

step("A. does the shock size matter? (default settings, 300 iterations)")
for s in (0.01, 0.05, 0.25)
    trial(@sprintf("US +%.0f pp on CHN", 100s), b0, s; max_iterations = 300)
end

step("B. does damping fix it? (25 pp, 300 iterations)")
for v in (0.10, 0.05, 0.02)
    trial(@sprintf("vfactor = %.2f", v), b0, 0.25; max_iterations = 300, vfactor = v)
end

step("C. does capping the elasticity tail fix it? (25 pp, default vfactor)")
for cap in (30.0, 20.0, 15.0)
    bc = recap(b0, cap)
    trial(@sprintf("theta capped at %.0f", cap), bc, 0.25; max_iterations = 300)
end

println("\ndone."); flush(stdout)
