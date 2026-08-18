# 05_emissions.jl — carbon accounts on the EMERGING baseline.
#
#     julia --project=. dev/emerging/05_emissions.jl
#
# Needs `04a_load.jl` to have cached the calibrated baseline and `05a_export_co2.jl` to have
# converted the satellite account to CSV. EMERGING is unusually well suited to Mahlkow & Wanner:
# it resolves all six fuels separately, where OECD ICIO merges crude oil with natural gas and
# gas distribution with electricity generation — collapses that make the Leontief fuel link
# unidentifiable.

using KITE, DataFrames, Printf, Statistics, Serialization, CSV

const DIR = joinpath(@__DIR__, "build", "2023")

step(s) = (println("\n", "="^74, "\n", s, "\n", "="^74); flush(stdout))

step("1. baseline and fossil taxonomy")
b = deserialize(joinpath(DIR, "baseline.jls"))
@printf("%d regions × %d sectors\n", b.N, b.J)

# EMERGING's energy sectors map straight onto the paper's taxonomy.
#   COAL, GAS   are extracted *and* burnt as extracted          -> P ∩ S
#   OIL         is extracted and refined before being burnt      -> P \ S
#   PETR, GASD  are burnt, Leontief-linked to their primary fuel -> S \ P
PRIMARY   = ["COAL", "OIL", "GAS"]
SECONDARY = Any["COAL", "GAS", "PETR" => "OIL", "GASD" => "GAS"]
println("primary   ", PRIMARY)
println("secondary ", SECONDARY)

step("2. read the CO₂ satellite account")
sat = CSV.read(joinpath(DIR, "co2_by_sector.csv"), DataFrame)
ci = Dict(c => i for (i, c) in enumerate(b.countries))
si = Dict(s => j for (j, s) in enumerate(b.sectors))
Z = zeros(b.N, b.J)                               # [region, burning industry], Mt CO₂
for row in eachrow(sat)
    Z[ci[row.country], si[row.sector]] += row.value
end
@printf("satellite: %d regions × %d industries, world total %.1f Mt CO₂\n", b.N, b.J, sum(Z))
top = sortperm(vec(sum(Z, dims = 1)), rev = true)[1:6]
println("largest emitting industries: ",
        join([@sprintf("%s %.0f", b.sectors[j], sum(Z[:, j])) for j in top], ", "))

step("3. calibrate carbon intensities")
# The satellite records CO₂ by *burning* industry, not by fuel, so the fuel split has to be
# inferred from which fuels each industry buys — the OECD TeCO2 situation as well.
model0 = MahlkowWanner2023(b; primary = PRIMARY, secondary = SECONDARY, resource_share = 0.6)
χ = emission_intensity_from_satellite(b, model0, Z)

# χ comes back in Mt CO₂ per USD-million of fuel absorbed, the units of the two inputs.
# Multiplying by 1000 puts it in kg CO₂ per USD, where physical carbon content is a benchmark.
for (i, s) in enumerate(model0.burnt_codes)
    j = findfirst(==(s), b.sectors)
    w = 1000 .* [χ[d, j] for d in 1:b.N if χ[d, j] > 0]
    @printf("  %-5s  median %6.1f  IQR %6.1f – %6.1f  kg CO₂ per USD  (%d regions)\n", s,
            median(w), quantile(w, 0.25), quantile(w, 0.75), length(w))
end
println("\n  physical benchmarks: coal ≈ 24, oil products ≈ 5.4, natural gas ≈ 5.3 kg CO₂/USD")
println("  refining and distribution margins put the processed fuels legitimately below theirs")

model = MahlkowWanner2023(b; primary = PRIMARY, secondary = SECONDARY,
                            resource_share = 0.6, emission_intensity = χ)
println("\n", model)

step("4. baseline accounts — the calibration check")
r0 = update_equilibrium(model, b; verbose = 0, vfactor = 0.05)
e0 = emissions(r0)
@printf("null scenario solved in %d iterations\n", r0.iterations)
@printf("world production  footprint %10.1f Mt   (satellite %10.1f Mt)\n",
        sum(e0.production), sum(Z))
@printf("world consumption footprint %10.1f Mt\n", sum(e0.consumption))
@printf("world extraction  footprint %10.1f Mt\n", sum(e0.extraction))
@printf("identities: production − consumption %.2e | production − extraction %.2e (relative)\n",
        (sum(e0.production) - sum(e0.consumption)) / sum(e0.production),
        (sum(e0.production) - sum(e0.extraction)) / sum(e0.production))

cmp = leftjoin(DataFrame(country = b.countries, satellite = vec(sum(Z, dims = 2))),
               e0[!, [:country, :production, :consumption, :extraction]], on = :country)
sort!(cmp, :satellite, rev = true)
println("\nlargest emitters (Mt CO₂):")
show(stdout, MIME("text/plain"), first(cmp, 15))

step("5. the three footprints diverge, which is the point")
cmp.net_import = cmp.consumption .- cmp.production
sort!(cmp, :net_import)
println("largest net *exporters* of embodied carbon (production ≫ consumption):")
show(stdout, MIME("text/plain"), first(cmp[!, [:country, :production, :consumption, :extraction, :net_import]], 8))
println("\n\nlargest net *importers* of embodied carbon:")
show(stdout, MIME("text/plain"), last(cmp[!, [:country, :production, :consumption, :extraction, :net_import]], 8))

step("6. counterfactual — United States raises tariffs on China by 25 pp")
sc = Scenario(b; label = "US +25pp on CHN")
set_tariff!(sc, b, 0.25; from = "CHN", to = "USA", mode = :add)
t = @elapsed r = update_equilibrium(model, b, sc; verbose = 0, vfactor = 0.05)
@printf("converged %s in %d iterations, %.1f s\n", r.converged, r.iterations, t)

e = emissions(r)
@printf("\nworld emissions %.1f -> %.1f Mt  (%+.3f%%)\n",
        sum(e.production), sum(e.production_new),
        100 * (sum(e.production_new) / sum(e.production) - 1))
@printf("identity after the shock: production − consumption %.2e (relative)\n",
        (sum(e.production_new) - sum(e.consumption_new)) / sum(e.production_new))

e.production_pct = 100 .* (e.production_change .- 1)
e.consumption_pct = 100 .* (e.consumption_change .- 1)
big = e[e.production .> 100, :]
sort!(big, :production_pct)
println("\nlargest falls in territorial emissions (of emitters above 100 Mt):")
show(stdout, MIME("text/plain"), first(big[!, [:country, :production, :production_pct, :consumption_pct]], 10))
println("\n\nlargest rises:")
show(stdout, MIME("text/plain"), last(big[!, [:country, :production, :production_pct, :consumption_pct]], 10))

step("7. where the change happens, by fuel")
ef = emissions(r; level = :fuel)
byfuel = combine(groupby(ef, :sector),
                 :production => sum => :base, :production_new => sum => :new)
byfuel.pct = 100 .* (byfuel.new ./ byfuel.base .- 1)
show(stdout, MIME("text/plain"), byfuel)

println("\n\ndone."); flush(stdout)
