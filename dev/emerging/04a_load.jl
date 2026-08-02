# 04a_load.jl — stage 1: load, calibrate, and gate on the null scenario.
# Progress is flushed as it happens, and the calibrated baseline is cached so the
# scenario script does not repeat the CSV parse.

using KITE, Printf, Statistics, Serialization

const DIR   = joinpath(@__DIR__, "build", "2023")
const CACHE = joinpath(DIR, "baseline.jls")

step(s) = (println("\n>>> ", s); flush(stdout))

step("reading CSVs and calibrating")
t = @elapsed b = read_baseline_csv(DIR)
@printf("loaded and calibrated in %.1f s\n", t); flush(stdout)
println(b); flush(stdout)

r = residuals(b)
@printf("\nresiduals: goods market %.3e | expenditure %.3e | income %.3e\n",
        r.goods_market, r.expenditure, r.income)
@printf("N = %d, J = %d | world value added %.0f mn USD | sum(D) = %.4g\n",
        b.N, b.J, sum(b.VA), sum(b.D))
@printf("theta: min %.2f median %.2f max %.2f | tariff: mean %.4f max %.3f\n",
        minimum(b.θ), median(b.θ), maximum(b.θ), mean(b.τ), maximum(b.τ))
flush(stdout)

step("null scenario (correctness gate)")
res0 = update_equilibrium(CaliendoParro2015(), b)
@printf("iterations %d | criterion %.3e | converged %s | %.2f s\n",
        res0.iterations, res0.criterion, res0.converged, res0.elapsed)
@printf("max |w-1| %.3e | max |P-1| %.3e | max |pi'-pi| %.3e\n",
        maximum(abs, res0.ŵ .- 1), maximum(abs, res0.P̂ .- 1),
        maximum(abs, res0.π′ .- b.π))
flush(stdout)

step("caching calibrated baseline")
serialize(CACHE, b)
@printf("wrote %s (%.0f MB)\n", CACHE, filesize(CACHE) / 2^20)
