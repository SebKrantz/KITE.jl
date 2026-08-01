# Build a shipped baseline from the long CSV interchange files.
#
#   1. Rscript dev/export_baseline_from_R.R <rds> dev/data/2022
#   2. julia --project=. dev/build_baseline_artifact.jl 2022
#
# Writes data/kite_baseline_<year>/ as dense little-endian Float64 blobs plus meta.toml.
# `load_baseline(year = <year>)` picks that directory up directly, so the baseline is usable
# straight away. To publish it, see dev/README.md.

using KITE, Printf

year = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 2022
src = length(ARGS) ≥ 2 ? ARGS[2] : joinpath(@__DIR__, "data", string(year))
root = dirname(@__DIR__)
dest = joinpath(root, "data", "kite_baseline_$(year)")

isdir(src) || error("no CSV directory at $src — run dev/export_baseline_from_R.R first.")

@info "reading and calibrating $src"
b = read_baseline_csv(src; verbose = 1)

res = residuals(b)
@printf("\nbaseline: %d countries × %d sectors\n", b.N, b.J)
@printf("residuals: goods %.2e  expenditure %.2e  income %.2e\n",
        res.goods_market, res.expenditure, res.income)
@printf("Σ VA = %.6g, Σ X = %.6g, Σ D = %.4g\n", sum(b.VA), sum(b.X), sum(b.D))

# The invariant that matters: a no-change scenario must reproduce the baseline exactly.
r = update_equilibrium(CaliendoParro2015(), b; verbose = 0, tolerance = 1e-12)
dev = max(maximum(abs, r.ŵ .- 1), maximum(abs, r.P̂ .- 1), maximum(abs, r.π′ .- b.π))
@printf("no-change scenario: %d iteration(s), max deviation %.2e\n", r.iterations, dev)
dev < 1e-9 || error("the calibrated baseline does not reproduce itself (deviation $dev).")

source = "OECD ICIO 2025 SML ($year); CEPII MAcMap-HS6 2019 tariffs; " *
         "Fontagné et al. (2022) trade elasticities"
write_baseline_binary(b, dest; year = year, source = source)

total = sum(filesize(joinpath(dest, f)) for f in readdir(dest))
@printf("\nwrote %s (%.1f MB)\n", dest, total / 1024^2)
@info "load it with `load_baseline(year = $year)`"
