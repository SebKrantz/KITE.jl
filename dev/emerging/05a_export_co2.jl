# 05a_export_co2.jl — EMERGING CO₂ satellite (.mat) -> long CSV.
#
#     julia dev/emerging/05a_export_co2.jl          # note: no --project
#
# The satellite is a MATLAB **v5** file, unlike the MRIO itself which is HDF5, so neither
# `rhdf5` nor the rest of the pipeline can read it. This runs in a throwaway environment with
# MAT.jl so that KITE.jl itself never takes a dependency on it, and writes a CSV the emissions
# script reads with the package's own CSV dependency.
#
# Output: dev/emerging/build/2023/co2_by_sector.csv, columns `country, sector, value` in Mt CO₂,
# where `sector` is the industry that *burns* the fuel.

using Pkg
Pkg.activate(temp = true)
Pkg.add(["MAT", "CSV", "DataFrames"]; io = devnull)
using MAT, CSV, DataFrames, Printf

const SRC   = "/Users/sebastiankrantz/Documents/Data/EMERGING/V2/EMERGING_CO2_2023.mat"
const BUILD = joinpath(@__DIR__, "build", "2023")
const NC_RAW, NS = 245, 133

co2 = matread(SRC)["CO2"]
size(co2, 1) == NC_RAW * NS || error("expected $(NC_RAW * NS) rows, got $(size(co2, 1))")
@printf("read %s, world total %.1f Mt CO₂ across %d fuel types\n",
        basename(SRC), sum(co2), size(co2, 2))

# rows are country-major, (c-1)*133 + s — the same order as the MRIO
tot = vec(sum(co2, dims = 2))

cm = CSV.read(joinpath(BUILD, "country_map.csv"), DataFrame)   # row, ISO3, kite
sm = CSV.read(joinpath(BUILD, "sector_map.csv"), DataFrame)    # pos, sector
nrow(cm) == NC_RAW && nrow(sm) == NS ||
    error("country_map.csv / sector_map.csv do not match the satellite dimensions")

out = DataFrame(country = repeat(cm.kite, inner = NS),
                sector = repeat(sm.sector, outer = NC_RAW),
                value = tot)
out = combine(groupby(out, [:country, :sector]), :value => sum => :value)   # pool into ROW
CSV.write(joinpath(BUILD, "co2_by_sector.csv"), out)
@printf("wrote co2_by_sector.csv: %d rows, %d regions, total %.1f Mt\n",
        nrow(out), length(unique(out.country)), sum(out.value))
