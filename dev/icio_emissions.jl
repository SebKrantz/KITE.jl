# icio_emissions.jl — carbon accounts on the 81 × 50 OECD ICIO baseline.
#
#     julia --project=. dev/icio_emissions.jl
#
# The EMERGING pipeline ships its own CO₂ satellite; ICIO does not, and no ICIO-aligned emissions
# file exists on this machine. So this script does two things:
#
#   * if `dev/data/2022/co2_by_sector.csv` is present it runs the real thing;
#   * otherwise it runs the identical code path on a synthetic satellite, purely to prove the
#     ICIO configuration solves and the accounting identities hold at ICIO's sector resolution.
#     Numbers from that branch are meaningless and the script says so, loudly, in every heading.
#
# To make it real, obtain CO₂ by country and ICIO industry — the OECD's own embodied-CO₂ data is
# the natural source, and IEA or EDGAR aggregated to the ICIO industry list also works — and save
# it as a long CSV with columns `country, sector, value`, using the baseline's own country and
# sector codes. Nothing else changes.
#
# Expect ICIO to be coarser than EMERGING, and for reasons of classification rather than code:
# `B06` merges crude petroleum with natural gas, and `D` merges gas distribution with electricity
# generation. Natural gas therefore gets no Leontief link of its own, gas burnt directly cannot
# be told apart from oil, and the distributed-gas channel disappears. Three of the model's
# mechanisms are simply not identified in this classification.

using KITE, DataFrames, Printf, Statistics, CSV, Random

const SAT = joinpath(@__DIR__, "data", "2022", "co2_by_sector.csv")

step(s) = (println("\n", "="^76, "\n", s, "\n", "="^76); flush(stdout))

step("1. baseline")
b = try
    load_baseline(year = 2022)
catch err
    @error "could not load the 2022 ICIO baseline; build it first with dev/build_baseline_artifact.jl"
    rethrow(err)
end
@printf("%d countries × %d sectors\n", b.N, b.J)

step("2. fossil taxonomy at ICIO resolution")
# B05 coal is extracted and burnt as extracted (P ∩ S).
# B06 is crude petroleum *and* natural gas together — extracted, and refined into C19.
# C19 coke and refined petroleum is burnt, Leontief-linked to B06.
# Gas distribution has no sector of its own: it is inside D, with electricity generation.
PRIMARY   = ["B05", "B06"]
SECONDARY = Any["B05", "C19" => "B06"]
for s in ["B05", "B06", "C19"]
    haskey(b.sector_index, s) || error("sector $s is absent from this baseline")
end
println("primary   ", PRIMARY)
println("secondary ", SECONDARY)
println("missing relative to EMERGING: natural gas as its own fuel, and gas distribution")

step("3. CO₂ satellite")
real_data = isfile(SAT)
Z = zeros(b.N, b.J)
"Scatter a long `country, sector, value` table into an `(N, J)` matrix, by label."
function scatter_satellite!(Z, b, sat)
    ci = Dict(c => i for (i, c) in enumerate(b.countries))
    si = Dict(s => j for (j, s) in enumerate(b.sectors))
    unknown = 0
    for row in eachrow(sat)
        if haskey(ci, row.country) && haskey(si, row.sector)
            Z[ci[row.country], si[row.sector]] += row.value
        else
            unknown += 1
        end
    end
    return unknown
end

if real_data
    sat = CSV.read(SAT, DataFrame)
    unknown = scatter_satellite!(Z, b, sat)
    @printf("read %s: %d rows, world total %.1f\n", basename(SAT), nrow(sat), sum(Z))
    unknown > 0 && @warn "$unknown rows had labels absent from the baseline and were dropped"
else
    @warn """
    No CO₂ satellite at $SAT — running on SYNTHETIC data.
    Everything below is a code-path check, not a result. See the header for what to supply."""
    # Fabricate something with the right shape: emissions roughly proportional to each industry's
    # fossil-fuel purchases, so the fit has a real signal to find, plus noise.
    rng = MersenneTwister(20260818)
    m0 = MahlkowWanner2023(b; primary = PRIMARY, secondary = SECONDARY)
    U, _ = KITE._absorption_matrices(b, m0)
    w = [2.0, 1.0]                                    # arbitrary per-fuel intensities
    for d in 1:b.N, k in 1:b.J
        Z[d, k] = max(0.0, sum(w[i] * U[d, i, k] for i in 1:length(m0.burnt)) *
                           (1 + 0.05 * randn(rng)))
    end
    @printf("synthetic satellite built, world total %.1f (arbitrary units)\n", sum(Z))
end
tag = real_data ? "" : "  [SYNTHETIC — not a result]"

step("4. calibrate and check the baseline" * tag)
m0 = MahlkowWanner2023(b; primary = PRIMARY, secondary = SECONDARY, resource_share = 0.6)
χ = emission_intensity_from_satellite(b, m0, Z)
model = MahlkowWanner2023(b; primary = PRIMARY, secondary = SECONDARY,
                            resource_share = 0.6, emission_intensity = χ)
println(model)

r0 = update_equilibrium(model, b; verbose = 0)
e0 = emissions(r0)
@printf("null scenario: %d iteration(s)\n", r0.iterations)
@printf("world production  %14.4f   (satellite %14.4f)\n", sum(e0.production), sum(Z))
@printf("world consumption %14.4f\n", sum(e0.consumption))
@printf("world extraction  %14.4f\n", sum(e0.extraction))
@printf("identities (relative): prod−cons %.2e | prod−extr %.2e\n",
        (sum(e0.production) - sum(e0.consumption)) / sum(e0.production),
        (sum(e0.production) - sum(e0.extraction)) / sum(e0.production))
@printf("country totals reproduce the satellite: max rel dev %.2e\n",
        maximum(abs, (e0.production .- vec(sum(Z, dims = 2))) ./
                     max.(vec(sum(Z, dims = 2)), 1e-12)))

step("5. a tariff counterfactual" * tag)
sc = Scenario(b; label = "US +25pp on CHN")
set_tariff!(sc, b, 0.25; from = "CHN", to = "USA", mode = :add)
t = @elapsed r = update_equilibrium(model, b, sc; verbose = 0)
e = emissions(r)
@printf("converged %s in %d iterations, %.1f s\n", r.converged, r.iterations, t)
@printf("world emissions %+.4f%%\n", 100 * (sum(e.production_new) / sum(e.production) - 1))
@printf("identity after the shock: %.2e\n",
        (sum(e.production_new) - sum(e.consumption_new)) / sum(e.production_new))

step("6. a carbon price" * tag)
sc2 = Scenario(b; label = "carbon price on the US")
# `price` is per unit of CO₂ in the satellite's units; 0.05 is illustrative here because the
# synthetic satellite has no physical scale.
set_carbon_price!(sc2, b, model, 0.05; country = "USA")
t2 = @elapsed r2 = update_equilibrium(model, b, sc2; verbose = 0, vfactor = 0.1,
                                      max_iterations = 5000)
e2 = emissions(r2)
@printf("converged %s in %d iterations, %.1f s\n", r2.converged, r2.iterations, t2)
iUS = b.country_index["USA"]
@printf("US territorial %+.3f%%, US consumption footprint %+.3f%%, world %+.3f%%\n",
        100 * (e2.production_change[iUS] - 1), 100 * (e2.consumption_change[iUS] - 1),
        100 * (sum(e2.production_new) / sum(e2.production) - 1))
@printf("realised ad-valorem wedge on B05 in the US: %.4f\n",
        r2.scenario.τ′[1, iUS, b.sector_index["B05"]] / b.τ[1, iUS, b.sector_index["B05"]])

println("\ndone.", real_data ? "" : "  (synthetic run — supply a satellite for real numbers)")
