# Tidy DataFrame assembly. Arrays stay dense internally; DataFrames appear only at the
# boundary, one row per country, country-sector, or origin-destination-sector.

"""
    results(r::KiteResult; level = :country, kwargs...) -> DataFrame

Tidy results at one of three levels of aggregation.

* `:country` — one row per country: trade, prices, income, production, welfare.
* `:sector` — one row per country-sector: prices, input costs, expenditure, production.
* `:bilateral` — one row per origin-destination-sector: trade shares and flows.

Extra keyword arguments are forwarded to [`country_results`](@ref), [`sector_results`](@ref) or
[`bilateral_results`](@ref).

# Examples
```julia
results(r)                                        # country level
results(r; level = :sector)
results(r; level = :bilateral, countries = "USA") # only flows touching the USA
```
"""
function results(r::KiteResult; level::Symbol = :country, kwargs...)
    level === :country && return country_results(r; kwargs...)
    level === :sector && return sector_results(r; kwargs...)
    level === :bilateral && return bilateral_results(r; kwargs...)
    error("level must be :country, :sector or :bilateral; got :$level.")
end

"""
    country_results(r::KiteResult) -> DataFrame

One row per country. Columns ending in `_new` are counterfactual levels, `_change` are ratios
of counterfactual to baseline.

Key columns: `welfare_change` (`Î/P̂`), `real_wage_change` (`ŵ/P̂`), `price_index_change`,
`wage_change`, `income`/`income_new`/`income_change`, `exports`/`imports` and their
counterparts, `tariff_revenue`, `export_subsidy_costs`, `production_total`,
`value_added`, `trade_balance`, `transfer`, and `weight` (baseline value-added share, for
aggregating welfare).
"""
function country_results(r::KiteResult)
    b = r.baseline
    ta = trade_aggregates(r)
    tr, tr_new = tariff_revenue(r)
    es, es_new = export_subsidy_costs(r)
    P̂ = price_index_change(r)
    prod_total = vec(sum(b.Y, dims = 2))
    prod_total_new = vec(sum(r.Y′, dims = 2))
    va_new = r.ŵ .* b.VA

    df = DataFrame(
        country = b.countries,
        wage_change = r.ŵ,
        price_index_change = P̂,
        real_wage_change = r.ŵ ./ P̂,
        welfare_change = (r.I′ ./ b.I) ./ P̂,
        income = b.I,
        income_new = r.I′,
        income_change = r.I′ ./ b.I,
        value_added = b.VA,
        value_added_new = va_new,
        value_added_change = va_new ./ b.VA,
        exports = ta.exports,
        exports_new = ta.exports_new,
        exports_change = _safe_ratio.(ta.exports_new, ta.exports),
        imports = ta.imports,
        imports_new = ta.imports_new,
        imports_change = _safe_ratio.(ta.imports_new, ta.imports),
        tariff_revenue = tr,
        tariff_revenue_new = tr_new,
        tariff_revenue_change = _safe_ratio.(tr_new, tr),
        export_subsidy_costs = es,
        export_subsidy_costs_new = es_new,
        export_subsidy_costs_change = _safe_ratio.(es_new, es),
        production_total = prod_total,
        production_total_new = prod_total_new,
        production_total_change = _safe_ratio.(prod_total_new, prod_total),
        production_total_real_new = prod_total_new ./ P̂,
        production_total_real_change = _safe_ratio.(prod_total_new ./ P̂, prod_total),
        trade_balance = b.D,
        trade_balance_new = r.D′,
        transfer = r.T′,
        weight = b.VA ./ sum(b.VA),
    )
    return df
end

"""
    sector_results(r::KiteResult) -> DataFrame

One row per country-sector: `price_change` (`P̂`), `input_cost_change` (`ĉ`), expenditure,
production (nominal and real), and the trade elasticity used.
"""
function sector_results(r::KiteResult)
    b = r.baseline
    P̂ = price_index_change(r)
    n = b.N * b.J
    country = Vector{String}(undef, n)
    sector = Vector{String}(undef, n)
    @inbounds for j in 1:b.J, d in 1:b.N
        i = d + b.N * (j - 1)
        country[i] = b.countries[d]
        sector[i] = b.sectors[j]
    end
    prod_real_new = r.Y′ ./ r.P̂

    return DataFrame(
        country = country,
        sector = sector,
        trade_elasticity = repeat(b.θ, inner = b.N),
        price_change = vec(r.P̂),
        input_cost_change = vec(r.ĉ),
        expenditure = vec(b.X),
        expenditure_new = vec(r.X′),
        expenditure_change = _safe_ratio.(vec(r.X′), vec(b.X)),
        production = vec(b.Y),
        production_new = vec(r.Y′),
        production_change = _safe_ratio.(vec(r.Y′), vec(b.Y)),
        production_real_new = vec(prod_real_new),
        production_real_change = _safe_ratio.(vec(prod_real_new), vec(b.Y)),
    )
end

"""
    bilateral_results(r::KiteResult; drop_zeros = true, countries = :all, sectors = :all) -> DataFrame

One row per origin-destination-sector: trade shares, gross flows (tariff-inclusive) and fob
flows (producer revenue), at baseline and counterfactual.

The full table has `N²J` rows — 328,050 for the shipped 2022 baseline — so `drop_zeros`
defaults to `true` and omits pairs with no baseline trade. `countries` filters on either side
of the flow.
"""
function bilateral_results(r::KiteResult; drop_zeros::Bool = true,
                           countries = :all, sectors = :all)
    b = r.baseline
    tf = trade_flows(r)
    ci = _countries(b, countries)
    si = _sectors(b, sectors)
    keep_country = falses(b.N)
    keep_country[ci] .= true

    origin = String[]
    destination = String[]
    sector = String[]
    share = Float64[]
    share_new = Float64[]
    flow = Float64[]
    flow_new = Float64[]
    fob = Float64[]
    fob_new = Float64[]
    tariff = Float64[]
    tariff_new = Float64[]

    @inbounds for j in si, d in 1:b.N, o in 1:b.N
        (keep_country[o] || keep_country[d]) || continue
        drop_zeros && b.π[o, d, j] == 0 && continue
        push!(origin, b.countries[o])
        push!(destination, b.countries[d])
        push!(sector, b.sectors[j])
        push!(share, b.π[o, d, j])
        push!(share_new, r.π′[o, d, j])
        push!(flow, tf.flow[o, d, j])
        push!(flow_new, tf.flow_new[o, d, j])
        push!(fob, tf.fob[o, d, j])
        push!(fob_new, tf.fob_new[o, d, j])
        push!(tariff, b.τ[o, d, j])
        push!(tariff_new, r.scenario.τ′[o, d, j])
    end

    return DataFrame(
        origin = origin,
        destination = destination,
        sector = sector,
        trade_share = share,
        trade_share_new = share_new,
        trade_share_change = _safe_ratio.(share_new, share),
        trade_flow = flow,
        trade_flow_new = flow_new,
        trade_flow_change = _safe_ratio.(flow_new, flow),
        trade_flow_fob = fob,
        trade_flow_fob_new = fob_new,
        trade_flow_fob_change = _safe_ratio.(fob_new, fob),
        tariff = tariff,
        tariff_new = tariff_new,
    )
end
