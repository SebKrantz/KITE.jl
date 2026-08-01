# Reading and writing baselines.
#
# Two formats:
#   * long CSV — the documented public interchange, one file per variable, the same column
#     names the R package uses, so an existing KITE dataset transfers unchanged;
#   * dense binary + meta.toml — what the shipped artifact holds, because parsing 830k CSV
#     rows costs about half a second while `read!` costs ten milliseconds.

const _CSV_TABLES = (
    ("trade_share",        (:origin, :destination, :sector)),
    ("intermediate_share", (:country, :input, :output)),
    ("factor_share",       (:country, :sector)),
    ("consumption_share",  (:country, :sector)),
    ("expenditure",        (:country, :sector)),
    ("value_added",        (:country,)),
    ("trade_balance",      (:country,)),
    ("tariff",             (:origin, :destination, :sector)),
    ("export_subsidy",     (:origin, :destination, :sector)),
    ("trade_elasticity",   (:sector,)),
)

const _REQUIRED_CSV = ("trade_share", "intermediate_share", "factor_share", "expenditure",
                       "trade_elasticity")

"""
    read_baseline_csv(dir; verbose = 1, kwargs...) -> KiteBaseline

Read a baseline from long-format CSV files in `dir` and [`calibrate`](@ref) it.

Expected files, each with its index columns plus `value` (the same schema as the R package's
initial conditions):

| file | columns |
|---|---|
| `trade_share.csv` | `origin, destination, sector, value` |
| `intermediate_share.csv` | `country, input, output, value` |
| `factor_share.csv` | `country, sector, value` |
| `expenditure.csv` | `country, sector, value` |
| `trade_elasticity.csv` | `sector, value` |
| `tariff.csv` *(optional)* | `origin, destination, sector, value` |
| `export_subsidy.csv` *(optional)* | `origin, destination, sector, value` |
| `consumption_share.csv` *(optional)* | `country, sector, value` |
| `value_added.csv`, `trade_balance.csv` *(optional)* | `country, value` |

Tables may be sparse: absent cells take a documented fill (`0` for shares and expenditure, `1`
for tariffs and export subsidies) and the number filled is reported. Extra keyword arguments go
to [`calibrate`](@ref).

# Examples
```julia
b = read_baseline_csv("dev/data/2022")
b = read_baseline_csv(dir; anchor = :value_added, verbose = 0)
```
"""
function read_baseline_csv(dir::AbstractString; verbose::Int = 1, kwargs...)
    isdir(dir) || error("read_baseline_csv: \"$dir\" is not a directory.")
    tables = Dict{String,Any}()
    for (name, keys) in _CSV_TABLES
        path = joinpath(dir, name * ".csv")
        isfile(path) || continue
        df = CSV.read(path, DataFrame)
        for k in keys
            hasproperty(df, k) ||
                error("read_baseline_csv: $name.csv is missing the column \"$k\".")
        end
        hasproperty(df, :value) ||
            error("read_baseline_csv: $name.csv is missing the column \"value\".")
        tables[name] = df
    end
    for k in _REQUIRED_CSV
        haskey(tables, k) || error("read_baseline_csv: \"$dir\" has no $k.csv.")
    end
    return baseline_from_long(tables; verbose = verbose, kwargs...)
end

"""
    write_baseline(b::KiteBaseline, dir; drop_zeros = true) -> String

Write `b` to long-format CSV files in `dir`, the inverse of [`read_baseline_csv`](@ref).
Zero trade shares are omitted by default, since [`read_baseline_csv`](@ref) fills them back in.
Returns `dir`.
"""
function write_baseline(b::KiteBaseline, dir::AbstractString; drop_zeros::Bool = true)
    mkpath(dir)
    CSV.write(joinpath(dir, "trade_share.csv"),
              _long3(b.π, b.countries, b.countries, b.sectors,
                     (:origin, :destination, :sector); drop = drop_zeros ? 0.0 : nothing))
    CSV.write(joinpath(dir, "intermediate_share.csv"),
              _long3(b.γ, b.countries, b.sectors, b.sectors, (:country, :input, :output);
                     drop = drop_zeros ? 0.0 : nothing))
    CSV.write(joinpath(dir, "tariff.csv"),
              _long3(b.τ, b.countries, b.countries, b.sectors,
                     (:origin, :destination, :sector); drop = drop_zeros ? 1.0 : nothing))
    CSV.write(joinpath(dir, "export_subsidy.csv"),
              _long3(b.ζ, b.countries, b.countries, b.sectors,
                     (:origin, :destination, :sector); drop = drop_zeros ? 1.0 : nothing))
    CSV.write(joinpath(dir, "factor_share.csv"),
              _long2(b.β, b.countries, b.sectors, (:country, :sector)))
    CSV.write(joinpath(dir, "consumption_share.csv"),
              _long2(b.α, b.countries, b.sectors, (:country, :sector)))
    CSV.write(joinpath(dir, "expenditure.csv"),
              _long2(b.X, b.countries, b.sectors, (:country, :sector)))
    CSV.write(joinpath(dir, "value_added.csv"),
              DataFrame(country = b.countries, value = b.VA))
    CSV.write(joinpath(dir, "trade_balance.csv"),
              DataFrame(country = b.countries, value = b.D))
    CSV.write(joinpath(dir, "trade_elasticity.csv"),
              DataFrame(sector = b.sectors, value = b.θ))
    return dir
end

function _long3(A, lab1, lab2, lab3, names; drop = nothing)
    c1 = String[]; c2 = String[]; c3 = String[]; v = Float64[]
    @inbounds for k in eachindex(lab3), j in eachindex(lab2), i in eachindex(lab1)
        x = A[i, j, k]
        drop !== nothing && x == drop && continue
        push!(c1, lab1[i]); push!(c2, lab2[j]); push!(c3, lab3[k]); push!(v, x)
    end
    return DataFrame(names[1] => c1, names[2] => c2, names[3] => c3, :value => v)
end

function _long2(A, lab1, lab2, names)
    c1 = String[]; c2 = String[]; v = Float64[]
    @inbounds for j in eachindex(lab2), i in eachindex(lab1)
        push!(c1, lab1[i]); push!(c2, lab2[j]); push!(v, A[i, j])
    end
    return DataFrame(names[1] => c1, names[2] => c2, :value => v)
end

# ── dense binary format ───────────────────────────────────────────────────────────────────

const _BIN_FIELDS = (("pi", :π, 3), ("gamma", :γ, 3), ("tau", :τ, 3), ("zeta", :ζ, 3),
                     ("alpha", :α, 2), ("beta", :β, 2), ("X", :X, 2), ("Y", :Y, 2),
                     ("theta", :θ, 1), ("I", :I, 1), ("R", :R, 1), ("VA", :VA, 1),
                     ("D", :D, 1))

"""
    write_baseline_binary(b::KiteBaseline, dir) -> String

Write `b` as column-major little-endian `Float64` blobs plus a `meta.toml` describing the
dimensions and layout. This is the format inside the shipped artifact; use
[`write_baseline`](@ref) for a human-readable, portable copy.
"""
function write_baseline_binary(b::KiteBaseline, dir::AbstractString; year::Int = 0,
                               source::AbstractString = "")
    mkpath(dir)
    for (file, field, _) in _BIN_FIELDS
        open(joinpath(dir, file * ".bin"), "w") do io
            write(io, htol.(vec(getfield(b, field))))
        end
    end
    open(joinpath(dir, "meta.toml"), "w") do io
        println(io, "# KITE.jl dense baseline")
        println(io, "[dataset]")
        println(io, "year = ", year)
        println(io, "source = ", repr(source))
        println(io, "layout = \"column-major Float64 little-endian\"")
        println(io, "[dims]")
        println(io, "N = ", b.N)
        println(io, "J = ", b.J)
    end
    open(joinpath(dir, "countries.txt"), "w") do io
        for c in b.countries; println(io, c); end
    end
    open(joinpath(dir, "sectors.txt"), "w") do io
        for s in b.sectors; println(io, s); end
    end
    return dir
end

"""
    read_baseline_binary(dir; check = true) -> KiteBaseline

Read a baseline written by [`write_baseline_binary`](@ref). The stored levels are already
model-consistent, so no calibration is performed; `check = false` skips the invariant test.
"""
function read_baseline_binary(dir::AbstractString; check::Bool = true)
    meta = TOML.parsefile(joinpath(dir, "meta.toml"))
    N = Int(meta["dims"]["N"])
    J = Int(meta["dims"]["J"])
    countries = readlines(joinpath(dir, "countries.txt"))
    sectors = readlines(joinpath(dir, "sectors.txt"))
    (length(countries) == N && length(sectors) == J) ||
        error("read_baseline_binary: label files do not match the declared dimensions.")

    dims = Dict(:π => (N, N, J), :γ => (N, J, J), :τ => (N, N, J), :ζ => (N, N, J),
                :α => (N, J), :β => (N, J), :X => (N, J), :Y => (N, J),
                :θ => (J,), :I => (N,), :R => (N,), :VA => (N,), :D => (N,))
    vals = Dict{Symbol,Any}()
    for (file, field, _) in _BIN_FIELDS
        d = dims[field]
        A = Array{Float64}(undef, d)
        open(joinpath(dir, file * ".bin"), "r") do io
            read!(io, A)
        end
        vals[field] = ltoh.(A)
    end

    return KiteBaseline(countries, sectors, vals[:π], vals[:γ], vals[:α], vals[:β], vals[:θ],
                        vals[:τ], vals[:ζ], vals[:X], vals[:Y], vals[:I], vals[:R],
                        vals[:VA], vals[:D]; check = check)
end

const _PKG_ROOT = dirname(@__DIR__)
const _ARTIFACTS_TOML = joinpath(_PKG_ROOT, "Artifacts.toml")

"""
    load_baseline(; year = 2022, check = true) -> KiteBaseline

Load a shipped, pre-calibrated baseline.

The 2022 vintage covers 81 countries and 50 sectors, built from the OECD ICIO 2025 SML tables,
CEPII MAcMap-HS6 2019 tariffs and the Fontagné et al. (2022) trade elasticities. It is
downloaded on first use as a lazy artifact and cached thereafter.

Resolution order: the `KITE_BASELINE_DIR` environment variable, then a locally built
`data/kite_baseline_<year>` directory in the package, then the registered artifact. The first
two make it possible to work with a freshly built baseline before it has been published.

# Examples
```julia
b = load_baseline()
r = update_equilibrium(CaliendoParro2015(), b)   # no-change benchmark: every hat is one
```
"""
function load_baseline(; year::Int = 2022, check::Bool = true)
    name = "kite_baseline_$(year)"

    env = get(ENV, "KITE_BASELINE_DIR", "")
    if !isempty(env)
        isdir(env) || error("load_baseline: KITE_BASELINE_DIR=\"$env\" is not a directory.")
        return read_baseline_binary(env; check = check)
    end

    local_dir = joinpath(_PKG_ROOT, "data", name)
    isdir(local_dir) && return read_baseline_binary(local_dir; check = check)

    isfile(_ARTIFACTS_TOML) || error(
        "load_baseline: no baseline for year $year. Build one with the scripts in `dev/`, " *
        "point KITE_BASELINE_DIR at it, or use `read_baseline_csv` with your own data.")
    hash = Artifacts.artifact_hash(name, _ARTIFACTS_TOML)
    hash === nothing && error(
        "load_baseline: Artifacts.toml has no entry \"$name\". Available vintages are listed " *
        "in the manual.")
    Artifacts.artifact_exists(hash) ||
        Artifacts.ensure_artifact_installed(name, _ARTIFACTS_TOML)
    return read_baseline_binary(Artifacts.artifact_path(hash); check = check)
end
