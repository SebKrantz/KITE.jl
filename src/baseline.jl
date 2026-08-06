# Long-format tables → dense arrays.
#
# Every value is placed by *label*, never by position. Positional reshape (`cast_variable` in
# the public R package) is fine when initial conditions are square; on incomplete grids it
# silently recycles values into the wrong cells. Here, missing cells take an explicit `fill`
# and the count is reported, duplicates are an error, and unknown labels are an error.

"""
    _scatter3(v1, v2, v3, values, lab1, lab2, lab3; fill = 0.0, name = "variable", verbose = 1)

Scatter long-format `values` keyed by the label triple `(v1, v2, v3)` into a dense
`(length(lab1), length(lab2), length(lab3))` array. Cells with no row take `fill`.
"""
function _scatter3(v1, v2, v3, values, lab1, lab2, lab3;
                   fill::Float64 = 0.0, name::AbstractString = "variable", verbose::Int = 1)
    i1 = Dict(l => i for (i, l) in enumerate(lab1))
    i2 = Dict(l => i for (i, l) in enumerate(lab2))
    i3 = Dict(l => i for (i, l) in enumerate(lab3))
    A = Base.fill(fill, length(lab1), length(lab2), length(lab3))
    seen = falses(size(A))
    n = length(values)
    (length(v1) == length(v2) == length(v3) == n) ||
        throw(DimensionMismatch("$name: key and value columns have different lengths."))
    @inbounds for r in 1:n
        a = get(i1, String(v1[r]), 0); a == 0 && error("$name: unknown label \"$(v1[r])\".")
        b = get(i2, String(v2[r]), 0); b == 0 && error("$name: unknown label \"$(v2[r])\".")
        c = get(i3, String(v3[r]), 0); c == 0 && error("$name: unknown label \"$(v3[r])\".")
        seen[a, b, c] && error("$name: duplicate entry for ($(v1[r]), $(v2[r]), $(v3[r])).")
        seen[a, b, c] = true
        A[a, b, c] = values[r]
    end
    _report_fill(seen, fill, name, verbose)
    return A
end

"""
    _scatter2(v1, v2, values, lab1, lab2; fill = 0.0, name = "variable", verbose = 1)

Two-dimensional counterpart of [`_scatter3`](@ref).
"""
function _scatter2(v1, v2, values, lab1, lab2;
                   fill::Float64 = 0.0, name::AbstractString = "variable", verbose::Int = 1)
    i1 = Dict(l => i for (i, l) in enumerate(lab1))
    i2 = Dict(l => i for (i, l) in enumerate(lab2))
    A = Base.fill(fill, length(lab1), length(lab2))
    seen = falses(size(A))
    n = length(values)
    (length(v1) == length(v2) == n) ||
        throw(DimensionMismatch("$name: key and value columns have different lengths."))
    @inbounds for r in 1:n
        a = get(i1, String(v1[r]), 0); a == 0 && error("$name: unknown label \"$(v1[r])\".")
        b = get(i2, String(v2[r]), 0); b == 0 && error("$name: unknown label \"$(v2[r])\".")
        seen[a, b] && error("$name: duplicate entry for ($(v1[r]), $(v2[r])).")
        seen[a, b] = true
        A[a, b] = values[r]
    end
    _report_fill(seen, fill, name, verbose)
    return A
end

"""
    _scatter1(v1, values, lab1; fill = 0.0, name = "variable", verbose = 1)

One-dimensional counterpart of [`_scatter3`](@ref).
"""
function _scatter1(v1, values, lab1;
                   fill::Float64 = 0.0, name::AbstractString = "variable", verbose::Int = 1)
    i1 = Dict(l => i for (i, l) in enumerate(lab1))
    A = Base.fill(fill, length(lab1))
    seen = falses(length(A))
    length(v1) == length(values) ||
        throw(DimensionMismatch("$name: key and value columns have different lengths."))
    @inbounds for r in eachindex(values)
        a = get(i1, String(v1[r]), 0); a == 0 && error("$name: unknown label \"$(v1[r])\".")
        seen[a] && error("$name: duplicate entry for $(v1[r]).")
        seen[a] = true
        A[a] = values[r]
    end
    _report_fill(seen, fill, name, verbose)
    return A
end

function _report_fill(seen, fill, name, verbose)
    missing_cells = count(!, seen)
    if missing_cells > 0 && verbose ≥ 1
        @info "$name: $missing_cells of $(length(seen)) cells absent from the input and set " *
              "to $fill."
    end
    return nothing
end

_unique_labels(cols...) = sort!(unique(String[String(x) for c in cols for x in c]))

"""
    baseline_from_long(tables::AbstractDict; kwargs...) -> KiteBaseline

Assemble and [`calibrate`](@ref) a baseline from long-format tables, keyed by the same names
the R package uses. Each value is a `NamedTuple`/`DataFrame`-like object with the index columns
plus `value`.

Required: `trade_share` (`origin`, `destination`, `sector`), `intermediate_share` (`country`,
`input`, `output`), `factor_share` (`country`, `sector`), `expenditure` (`country`, `sector`),
`trade_elasticity` (`sector`). Optional: `tariff`, `export_subsidy` (default no policy),
`consumption_share` (only used with `final_demand = :given`), `value_added`, `trade_balance`
(only used with `anchor = :value_added`).

Extra keyword arguments are forwarded to [`calibrate`](@ref).
"""
function baseline_from_long(tables::AbstractDict; verbose::Int = 1, kwargs...)
    for k in ("trade_share", "intermediate_share", "factor_share", "expenditure",
              "trade_elasticity")
        haskey(tables, k) || error("baseline_from_long: missing required table \"$k\".")
    end
    ts = tables["trade_share"]
    is = tables["intermediate_share"]
    fs = tables["factor_share"]

    countries = _unique_labels(_col(ts, :origin), _col(ts, :destination), _col(is, :country),
                               _col(fs, :country))
    sectors = _unique_labels(_col(ts, :sector), _col(is, :input), _col(is, :output),
                             _col(fs, :sector))
    N, J = length(countries), length(sectors)

    π = _scatter3(_col(ts, :origin), _col(ts, :destination), _col(ts, :sector),
                  _col(ts, :value), countries, countries, sectors;
                  fill = 0.0, name = "trade_share", verbose = verbose)
    γ = _scatter3(_col(is, :country), _col(is, :input), _col(is, :output),
                  _col(is, :value), countries, sectors, sectors;
                  fill = 0.0, name = "intermediate_share", verbose = verbose)
    β = _scatter2(_col(fs, :country), _col(fs, :sector), _col(fs, :value),
                  countries, sectors; fill = 1 - 1e-4, name = "factor_share", verbose = verbose)

    ex = tables["expenditure"]
    X = _scatter2(_col(ex, :country), _col(ex, :sector), _col(ex, :value),
                  countries, sectors; fill = 0.0, name = "expenditure", verbose = verbose)

    te = tables["trade_elasticity"]
    θ = _scatter1(_col(te, :sector), _col(te, :value), sectors;
                  fill = 1.0, name = "trade_elasticity", verbose = verbose)

    τ = if haskey(tables, "tariff")
        t = tables["tariff"]
        _scatter3(_col(t, :origin), _col(t, :destination), _col(t, :sector), _col(t, :value),
                  countries, countries, sectors; fill = 1.0, name = "tariff", verbose = verbose)
    else
        ones(N, N, J)
    end
    ζ = if haskey(tables, "export_subsidy")
        t = tables["export_subsidy"]
        _scatter3(_col(t, :origin), _col(t, :destination), _col(t, :sector), _col(t, :value),
                  countries, countries, sectors; fill = 1.0, name = "export_subsidy",
                  verbose = verbose)
    else
        ones(N, N, J)
    end

    α = if haskey(tables, "consumption_share")
        cs = tables["consumption_share"]
        _scatter2(_col(cs, :country), _col(cs, :sector), _col(cs, :value), countries, sectors;
                  fill = 0.0, name = "consumption_share", verbose = verbose)
    else
        nothing
    end
    VA = haskey(tables, "value_added") ?
        _scatter1(_col(tables["value_added"], :country), _col(tables["value_added"], :value),
                  countries; fill = 0.0, name = "value_added", verbose = verbose) : nothing
    D = haskey(tables, "trade_balance") ?
        _scatter1(_col(tables["trade_balance"], :country), _col(tables["trade_balance"], :value),
                  countries; fill = 0.0, name = "trade_balance", verbose = verbose) : nothing

    return calibrate(; countries = countries, sectors = sectors, π = π, γ = γ, β = β, θ = θ,
                       X = X, α = α, VA = VA, D = D, τ = τ, ζ = ζ, verbose = verbose, kwargs...)
end

# Column accessor that works for DataFrames and NamedTuples alike.
function _col(t, name::Symbol)
    if t isa AbstractDict
        haskey(t, name) && return t[name]
        haskey(t, String(name)) && return t[String(name)]
        error("table is missing column \"$name\".")
    end
    hasproperty(t, name) || error("table is missing column \"$name\".")
    return getproperty(t, name)
end
