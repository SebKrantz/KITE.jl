# Scenario construction helpers.
#
# Selectors accept `:all`, a single label, or a vector of labels, and are resolved to integer
# indices once against the baseline's label dictionaries.

function _resolve(sel, index::Dict{String,Int}, n::Int, what::AbstractString)
    sel === :all && return 1:n
    if sel isa AbstractString
        haskey(index, sel) || error("unknown $what \"$sel\".")
        return index[sel]:index[sel]
    end
    if sel isa Integer
        1 ≤ sel ≤ n || error("$what index $sel is out of bounds (1:$n).")
        return sel:sel
    end
    if sel isa AbstractVector
        isempty(sel) && return 1:0
        if eltype(sel) <: Integer
            all(i -> 1 ≤ i ≤ n, sel) || error("a $what index is out of bounds (1:$n).")
            return collect(Int, sel)
        end
        out = Vector{Int}(undef, length(sel))
        for (i, s) in enumerate(sel)
            key = String(s)
            haskey(index, key) || error("unknown $what \"$key\".")
            out[i] = index[key]
        end
        return out
    end
    error("$what selector must be :all, a label, or a vector of labels; got $(typeof(sel)).")
end

_countries(b, sel) = _resolve(sel, b.country_index, b.N, "country")
_sectors(b, sel) = _resolve(sel, b.sector_index, b.J, "sector")

function _apply!(dest, rows, cols, sheets, value, mode::Symbol, name::AbstractString)
    mode in (:set, :multiply, :add) ||
        error("mode must be :set, :multiply or :add; got :$mode.")
    @inbounds for j in sheets, d in cols, o in rows
        if mode === :set
            dest[o, d, j] = value
        elseif mode === :multiply
            dest[o, d, j] *= value
        else
            dest[o, d, j] += value
        end
    end
    return dest
end

"""
    set_tariff!(sc, b, value; from = :all, to = :all, sector = :all, mode = :set)

Set counterfactual tariffs on flows from `from` to `to` in `sector`.

`value` is a **multiplier**: `1.25` is a 25% ad-valorem tariff, `1.0` is free trade. `mode`
selects how it is combined with what is already there — `:set` overwrites, `:multiply` scales
the existing multiplier, `:add` adds to the ad-valorem rate.

Selectors take `:all`, a country/sector code, or a vector of codes.

# Examples
```julia
set_tariff!(sc, b, 1.25; from = "CHN", to = "USA")             # 25% across all sectors
set_tariff!(sc, b, 1.10; from = "CHN", to = "USA", sector = "C26", mode = :multiply)
set_tariff!(sc, b, 1.05; from = ["DEU", "FRA"], to = "GBR")
```
"""
function set_tariff!(sc::Scenario, b::KiteBaseline, value::Real;
                     from = :all, to = :all, sector = :all, mode::Symbol = :set)
    value > 0 || error("tariff value must be positive; got $value.")
    _apply!(sc.τ′, _countries(b, from), _countries(b, to), _sectors(b, sector),
            Float64(value), mode, "tariff")
    return sc
end

"""
    set_ntb!(sc, b, value; from = :all, to = :all, sector = :all, mode = :set)

Set the non-tariff-barrier **change** `κ̂` on the selected flows. `1.0` is no change, values
above one raise iceberg trade costs. Sanctions shocks estimated from a gravity regression enter
here, as `exp(-δ̂ / θ)`.

# Examples
```julia
set_ntb!(sc, b, 1.5; from = "RUS", to = :all)   # 50% higher trade costs on Russian exports
set_ntb!(sc, b, 1e6; from = "RUS", to = "USA")  # prohibitive: an embargo
```
"""
function set_ntb!(sc::Scenario, b::KiteBaseline, value::Real;
                  from = :all, to = :all, sector = :all, mode::Symbol = :set)
    value > 0 || error("ntb value must be positive; got $value.")
    _apply!(sc.κ̂, _countries(b, from), _countries(b, to), _sectors(b, sector),
            Float64(value), mode, "ntb")
    return sc
end

"""
    set_export_subsidy!(sc, b, value; from = :all, to = :all, sector = :all, mode = :set)

Set the counterfactual export tax (`> 1`) or subsidy (`< 1`) multiplier `ζ′` on the selected
flows, where the origin `from` is the country levying or paying it.
"""
function set_export_subsidy!(sc::Scenario, b::KiteBaseline, value::Real;
                             from = :all, to = :all, sector = :all, mode::Symbol = :set)
    value > 0 || error("export subsidy value must be positive; got $value.")
    _apply!(sc.ζ′, _countries(b, from), _countries(b, to), _sectors(b, sector),
            Float64(value), mode, "export subsidy")
    return sc
end

"""
    set_productivity!(sc, b, value; country = :all, sector = :all, mode = :set)

Set the exogenous productivity change `ẑ`. Values above one lower input costs.
"""
function set_productivity!(sc::Scenario, b::KiteBaseline, value::Real;
                           country = :all, sector = :all, mode::Symbol = :set)
    value > 0 || error("productivity value must be positive; got $value.")
    mode in (:set, :multiply, :add) ||
        error("mode must be :set, :multiply or :add; got :$mode.")
    @inbounds for j in _sectors(b, sector), d in _countries(b, country)
        if mode === :set
            sc.ẑ[d, j] = value
        elseif mode === :multiply
            sc.ẑ[d, j] *= value
        else
            sc.ẑ[d, j] += value
        end
    end
    return sc
end

"""
    set_population!(sc, b, value; country = :all, mode = :set)

Set the exogenous labour-supply change `L̂`.
"""
function set_population!(sc::Scenario, b::KiteBaseline, value::Real;
                         country = :all, mode::Symbol = :set)
    value > 0 || error("population value must be positive; got $value.")
    mode in (:set, :multiply, :add) ||
        error("mode must be :set, :multiply or :add; got :$mode.")
    @inbounds for d in _countries(b, country)
        if mode === :set
            sc.L̂[d] = value
        elseif mode === :multiply
            sc.L̂[d] *= value
        else
            sc.L̂[d] += value
        end
    end
    return sc
end

"""
    set_coalition!(sc, b, countries)

Define the sanctioning coalition for [`ChowdhryHinzKaminWanner2022`](@ref). Members share the
welfare cost equally through transfers that net to zero within the coalition. Replaces any
previous membership.

# Examples
```julia
set_coalition!(sc, b, ["USA", "DEU", "FRA", "GBR"])
```
"""
function set_coalition!(sc::Scenario, b::KiteBaseline, countries)
    fill!(sc.coalition, false)
    for d in _countries(b, countries isa AbstractString ? [countries] : countries)
        sc.coalition[d] = true
    end
    return sc
end
