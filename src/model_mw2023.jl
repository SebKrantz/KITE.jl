# Mahlkow & Wanner (2023), "The Carbon Footprint of Global Trade Imbalances".
# KITE whitepaper §2.4, equations (23)–(29).
#
# Two departures from Caliendo & Parro:
#
#   * Primary fossil-fuel sectors (crude oil, coal, gas extraction) use a sector-specific
#     natural resource in fixed supply alongside labour and intermediates. Its rental price
#     `p̂^r` is an additional endogenous factor price, and resource rents are part of income.
#
#   * Secondary fossil-fuel sectors (refined petroleum, say) combine their *complementary*
#     primary fuel with everything else in **Leontief** rather than Cobb-Douglas fashion. Cost
#     shares therefore move with relative prices, which is what lets fuel use — and so
#     emissions — respond to trade shocks rather than being pinned by a fixed exponent.
#
# The whitepaper reproduces the model without tariffs or export subsidies. KITE.jl keeps them,
# since the extension is mechanical and the shipped baseline has tariffs; with no fossil sectors
# designated the model then reduces *exactly* to `CaliendoParro2015`, which is the headline test.
#
# Sign note: whitepaper eq. (28) writes income with `+ D_d′`, where every other equation in the
# paper (and the KITE data convention) treats `D` as a surplus subtracted from income. We follow
# the package-wide convention, `− D`.

"""
    MahlkowWanner2023(b::KiteBaseline; primary = String[], secondary = Pair[],
                      resource_share = 0.5)

Carbon/energy extension of [`CaliendoParro2015`](@ref) after Mahlkow & Wanner (2023).

# Keyword Arguments
- `primary`: sector codes of primary fossil fuels — those extracting a natural resource in fixed
  supply. Each earns a rent whose price is solved for alongside wages.
- `secondary`: pairs `"secondary" => "primary"` linking each secondary fuel to its complementary
  primary fuel, e.g. `["C19" => "B06"]` for refined petroleum and crude oil. Secondary sectors
  use that fuel in fixed proportion (Leontief) to the rest of their input bundle.
- `resource_share`: the share of a primary sector's **value added** accruing to the natural
  resource rather than to labour, in `(0, 1)`. Accepts a scalar, a `Dict` keyed by sector code,
  or a full `(N, J)` matrix of shares. The remaining value added is labour income.

With `primary` and `secondary` both empty this solves exactly the same equilibrium as
[`CaliendoParro2015`](@ref).

# Examples
```julia
model = MahlkowWanner2023(b;
    primary = ["B05", "B06"],               # coal, oil and gas extraction
    secondary = ["C19" => "B06"],           # refined petroleum ← crude oil
    resource_share = 0.6)

r = update_equilibrium(model, b, sc)
fossil_use(r)                                # real secondary-fuel use, the emissions driver
```

See also [`fossil_use`](@ref), [`resource_price_change`](@ref).
"""
struct MahlkowWanner2023 <: KiteModel
    primary::Vector{Int}
    secondary::Vector{Int}
    complement::Vector{Int}     # complement[i] is the primary sector index for secondary[i]
    γr::Matrix{Float64}         # (N, J) resource cost share of gross output; 0 off `primary`
    γl::Matrix{Float64}         # (N, J) labour cost share of gross output, = β − γr
    primary_codes::Vector{String}
    secondary_codes::Vector{String}
end

function MahlkowWanner2023(b::KiteBaseline; primary = String[], secondary = Pair[],
                           resource_share = 0.5)
    prim = [_sector_index(b, s) for s in primary]
    allunique(prim) || error("primary sectors must be distinct.")

    sec = Int[]
    comp = Int[]
    for pr in secondary
        pr isa Pair || error("secondary must be given as \"secondary\" => \"primary\" pairs; " *
                             "got $(typeof(pr)).")
        s = _sector_index(b, first(pr))
        q = _sector_index(b, last(pr))
        s in prim && error("sector \"$(first(pr))\" cannot be both primary and secondary.")
        push!(sec, s)
        push!(comp, q)
    end
    allunique(sec) || error("secondary sectors must be distinct.")

    γr = zeros(b.N, b.J)
    if !isempty(prim)
        share = _resource_share_matrix(b, resource_share, prim)
        for j in prim, d in 1:b.N
            0 ≤ share[d, j] < 1 ||
                error("resource_share must lie in [0, 1); got $(share[d, j]) for " *
                      "$(b.countries[d]), $(b.sectors[j]).")
            γr[d, j] = share[d, j] * b.β[d, j]
        end
    end
    γl = b.β .- γr
    all(>(0), γl) || error("resource_share leaves a non-positive labour share in some " *
                           "primary sector; lower it.")

    return MahlkowWanner2023(prim, sec, comp, γr, γl,
                             String[b.sectors[j] for j in prim],
                             String[b.sectors[j] for j in sec])
end

function _sector_index(b::KiteBaseline, s)
    key = String(s)
    haskey(b.sector_index, key) || error("unknown sector \"$key\".")
    return b.sector_index[key]
end

_resource_share_matrix(b, x::Real, prim) = fill(Float64(x), b.N, b.J)
function _resource_share_matrix(b, x::AbstractMatrix, prim)
    _check_size(x, (b.N, b.J), "resource_share")
    return Float64.(x)
end
function _resource_share_matrix(b, x::AbstractDict, prim)
    m = zeros(b.N, b.J)
    for j in prim
        code = b.sectors[j]
        haskey(x, code) ||
            error("resource_share has no entry for primary sector \"$code\".")
        m[:, j] .= Float64(x[code])
    end
    return m
end

_model_name(::MahlkowWanner2023) = "mahlkow_wanner_2023"

function Base.show(io::IO, m::MahlkowWanner2023)
    print(io, "MahlkowWanner2023(", length(m.primary), " primary, ",
          length(m.secondary), " secondary fossil sectors)")
end

# ── model state ───────────────────────────────────────────────────────────────────────────

mutable struct _MWState
    primary::Vector{Int}
    secondary::Vector{Int}
    complement::Vector{Int}
    is_secondary::Vector{Bool}      # (J)
    sec_slot::Vector{Int}           # (J) position in `secondary`, 0 if not secondary
    γr::Matrix{Float64}             # (N, J)
    γl::Matrix{Float64}             # (N, J)
    L::Vector{Float64}              # (N) baseline labour income  Σ_j γl·Y
    Rinc::Matrix{Float64}           # (N, n_p) baseline resource rents  γr·Y
    p̂::Matrix{Float64}              # (N, n_p) resource price change
    labour′::Vector{Float64}        # (N) labour payments implied by the new output
    rent′::Matrix{Float64}          # (N, n_p) resource rents implied by the new output
    a::Matrix{Float64}              # (N, n_s) Leontief share on the complementary fuel
    γ̂_fuel::Matrix{Float64}         # (N, n_s) change in that share
    γ̂_rest::Matrix{Float64}         # (N, n_s) change in every other share of the same sector
end

function _model_state(m::MahlkowWanner2023, b::KiteBaseline, sc::Scenario)
    N, J = b.N, b.J
    n_p, n_s = length(m.primary), length(m.secondary)

    is_secondary = falses(J)
    sec_slot = zeros(Int, J)
    for (i, s) in enumerate(m.secondary)
        is_secondary[s] = true
        sec_slot[s] = i
    end

    L = vec(sum(m.γl .* b.Y, dims = 2))
    all(>(0), L) || error("MahlkowWanner2023: some country has no labour income at the " *
                          "baseline; check resource_share.")

    Rinc = Matrix{Float64}(undef, N, n_p)
    for (i, p) in enumerate(m.primary)
        @views @. Rinc[:, i] = m.γr[:, p] * b.Y[:, p]
    end
    any(iszero, Rinc) && @info "MahlkowWanner2023: some country earns no rent in a primary " *
                               "sector at the baseline; its resource price is held fixed."

    a = Matrix{Float64}(undef, N, n_s)
    for (i, s) in enumerate(m.secondary)
        q = m.complement[i]
        @views @. a[:, i] = b.input_share[:, q, s]
    end
    any(x -> x ≥ 1 - 1e-12, a) &&
        error("MahlkowWanner2023: a secondary sector spends its entire input budget on its " *
              "complementary fuel, leaving no Cobb-Douglas bundle to substitute into.")

    return _MWState(m.primary, m.secondary, m.complement, is_secondary, sec_slot,
                    m.γr, m.γl, L, Rinc, ones(N, n_p), zeros(N), zeros(N, n_p),
                    a, ones(N, n_s), ones(N, n_s))
end

# ── equations ─────────────────────────────────────────────────────────────────────────────

"""
    _input_cost!(::MahlkowWanner2023, ws, b, sc)

Whitepaper equations (23) and (24). Ordinary and primary sectors keep the Cobb-Douglas form,
primary sectors adding the resource price:

    ĉ[d,j] = exp( γr·log p̂ʳ[d,j] + γl·log ŵ[d] + Σ_k input_share[d,k,j]·log P̂[d,k] )

A secondary fuel `s` with complementary primary fuel `q` is Leontief between that fuel and a
Cobb-Douglas bundle of everything else, so its cost is an *arithmetic* average and its cost
shares move:

    B       = exp( [γl[d,s]·log ŵ[d] + Σ_{k≠q} input_share[d,k,s]·log P̂[d,k]] / (1 − a) )
    ĉ[d,s]  = a·P̂[d,q] + (1 − a)·B
    γ̂_fuel  = P̂[d,q] / ĉ[d,s],      γ̂_rest = B / ĉ[d,s]

Productivity shocks are applied after the cost shares are read off, since a Hicks-neutral gain
scales every input requirement equally and so leaves shares unchanged.
"""
function _input_cost!(m::MahlkowWanner2023, ws::_Workspace, b::KiteBaseline, sc::Scenario)
    N, J = ws.N, ws.J
    st = ws.ext
    @inbounds @. ws.logP̂ = log(ws.P̂)

    @inbounds for j in 1:J
        cj = view(ws.ĉ, :, j)
        if !st.is_secondary[j]
            fill!(cj, 0.0)
            for k in 1:J
                @views @. cj += b.input_share[:, k, j] * ws.logP̂[:, k]
            end
            @views @. cj += st.γl[:, j] * log(ws.ŵ)
            for (i, p) in enumerate(st.primary)
                p == j || continue
                @views @. cj += st.γr[:, j] * log(st.p̂[:, i])
            end
            @. cj = exp(cj)
        else
            i = st.sec_slot[j]
            q = st.complement[i]
            for d in 1:N
                acc = st.γl[d, j] * log(ws.ŵ[d])
                for k in 1:J
                    k == q && continue
                    acc += b.input_share[d, k, j] * ws.logP̂[d, k]
                end
                ad = st.a[d, i]
                B = exp(acc / (1 - ad))
                Pq = ws.P̂[d, q]
                c = ad * Pq + (1 - ad) * B
                ws.ĉ[d, j] = c
                st.γ̂_fuel[d, i] = Pq / c
                st.γ̂_rest[d, i] = B / c
            end
        end
    end

    ws.has_productivity && @inbounds @. ws.ĉ /= sc.ẑ
    return ws
end

"""
    _intermediate_demand_model!(::MahlkowWanner2023, ws, b)

`ID[d,k] = Σ_j input_share[d,k,j]·Y′[d,j]`, with each secondary sector's column rescaled by the
cost-share changes of whitepaper equation (24): the complementary-fuel row by `γ̂_fuel`, every
other row by `γ̂_rest`.
"""
function _intermediate_demand_model!(m::MahlkowWanner2023, ws::_Workspace, b::KiteBaseline)
    N, J = ws.N, ws.J
    st = ws.ext
    ID = ws.ID
    fill!(ID, 0.0)
    @inbounds for j in 1:J
        if !st.is_secondary[j]
            for k in 1:J
                @views @. ID[:, k] += b.input_share[:, k, j] * ws.Y′[:, j]
            end
        else
            i = st.sec_slot[j]
            q = st.complement[i]
            for k in 1:J
                g = k == q ? view(st.γ̂_fuel, :, i) : view(st.γ̂_rest, :, i)
                @views @. ID[:, k] += g * b.input_share[:, k, j] * ws.Y′[:, j]
            end
        end
    end
    return ID
end

"""
    _factor_income!(::MahlkowWanner2023, ws, b, sc)

Whitepaper equation (28): labour income plus rents on the fixed natural resources,
`I_factor[d] = L̂[d]·ŵ[d]·L[d] + Σ_p p̂ʳ[d,p]·(p^r R)[d,p]`.
"""
function _factor_income!(m::MahlkowWanner2023, ws::_Workspace, b::KiteBaseline, sc::Scenario)
    st = ws.ext
    @inbounds for d in 1:ws.N
        lw = ws.has_population ? sc.L̂[d] * ws.ŵ[d] : ws.ŵ[d]
        acc = lw * st.L[d]
        for i in eachindex(st.primary)
            acc += st.p̂[d, i] * st.Rinc[d, i]
        end
        ws.factor_income[d] = acc
    end
    return ws
end

"""
    _value_added!(::MahlkowWanner2023, ws, b)

Factor payments implied by the new output — the numerators of whitepaper equations (29a) and
(29b). Secondary sectors are weighted by their revised cost shares.
"""
function _value_added!(m::MahlkowWanner2023, ws::_Workspace, b::KiteBaseline)
    N, J = ws.N, ws.J
    st = ws.ext

    fill!(st.labour′, 0.0)
    @inbounds for j in 1:J
        if !st.is_secondary[j]
            @views @. st.labour′ += st.γl[:, j] * ws.Y′[:, j]
        else
            i = st.sec_slot[j]
            @views @. st.labour′ += st.γ̂_rest[:, i] * st.γl[:, j] * ws.Y′[:, j]
        end
    end
    @inbounds for (i, p) in enumerate(st.primary)
        @views @. st.rent′[:, i] = st.γr[:, p] * ws.Y′[:, p]
    end

    copyto!(ws.VA′, st.labour′)
    @inbounds for i in eachindex(st.primary)
        @views @. ws.VA′ += st.rent′[:, i]
    end
    return ws
end

_transfers!(::MahlkowWanner2023, ws::_Workspace, b::KiteBaseline, sc::Scenario) = ws

# Primary sectors leave the block linear — the resource price is fixed during the inner loop.
# Secondary sectors do not: their Leontief nest rescales the input coefficients by γ̂.
_linear_expenditure(::MahlkowWanner2023, ws::_Workspace) = isempty(ws.ext.secondary)

_check_scenario(::MahlkowWanner2023, b::KiteBaseline, sc::Scenario) = nothing

"""
    _wage_update!(::MahlkowWanner2023, ws, b, sc, settings)

Whitepaper equations (29a) and (29b): the resource rental price clears the fixed resource
supply, and the wage clears the labour market. Both are updated in logs and dampened by
`vfactor`, as in [`CaliendoParro2015`](@ref).
"""
function _wage_update!(m::MahlkowWanner2023, ws::_Workspace, b::KiteBaseline, sc::Scenario,
                       settings::SolverSettings)
    v = settings.vfactor
    st = ws.ext

    @inbounds for d in 1:ws.N
        st.L[d] > 0 || continue
        target = st.labour′[d] / st.L[d]
        ws.has_population && (target /= sc.L̂[d])
        target > 0 || continue
        ws.ŵ[d] = exp((1 - v) * log(ws.ŵ[d]) + v * log(target))
    end

    @inbounds for i in eachindex(st.primary), d in 1:ws.N
        st.Rinc[d, i] > 0 || continue
        target = st.rent′[d, i] / st.Rinc[d, i]
        target > 0 || continue
        st.p̂[d, i] = exp((1 - v) * log(st.p̂[d, i]) + v * log(target))
    end
    return ws
end

"""
    _normalise_wage!(::MahlkowWanner2023, ws, b, num_index)

Impose the numéraire on *every* factor price. Scaling wages alone would leave resource rents
behind and change relative prices, so `ŵ` and `p̂ʳ` move together.
"""
function _normalise_wage!(m::MahlkowWanner2023, ws::_Workspace, b::KiteBaseline,
                          num_index::Int)
    st = ws.ext
    if num_index == 0
        s_new = sum(ws.VA′)
        s_new > 0 || return ws
        scale = sum(b.VA) / s_new
    else
        w = ws.ŵ[num_index]
        w > 0 || return ws
        scale = 1 / w
    end
    isfinite(scale) && scale > 0 || return ws
    @inbounds @. ws.ŵ *= scale
    @inbounds @. st.p̂ *= scale
    @inbounds @. ws.VA′ *= scale
    return ws
end

# ── results helpers ───────────────────────────────────────────────────────────────────────

"""
    resource_price_change(r::KiteResult{MahlkowWanner2023}) -> DataFrame

Change in the rental price of each primary fossil-fuel resource, by country and sector.
"""
function resource_price_change(r::KiteResult{MahlkowWanner2023})
    m = r.model
    isempty(m.primary) && return DataFrame(country = String[], sector = String[], value = Float64[])
    n = r.baseline.N
    return DataFrame(
        country = repeat(r.baseline.countries, length(m.primary)),
        sector = repeat(m.primary_codes, inner = n),
        value = vec(r.ext.resource_price),
    )
end

"""
    _result_extras(::MahlkowWanner2023, ws)

Carries out the resource rental prices and the Leontief cost-share changes.

`fuel_cost_share_change` is `γ̂` on the complementary primary fuel and `other_cost_share_change`
is `γ̂` on everything else in the same secondary sector. Because the sector's unit cost is a
weighted average of the two components' prices, these always straddle one: whichever input got
relatively dearer takes a larger share of the bill, since the Leontief nest fixes the quantity
ratio and rules out substituting away from it.
"""
_result_extras(::MahlkowWanner2023, ws::_Workspace) =
    (resource_price = copy(ws.ext.p̂),
     fuel_cost_share_change = copy(ws.ext.γ̂_fuel),
     other_cost_share_change = copy(ws.ext.γ̂_rest))

"""
    fossil_use(r::KiteResult{MahlkowWanner2023}) -> DataFrame

Real use of each secondary fossil fuel, by country — the quantity that drives emissions in
Mahlkow & Wanner (2023).

Nominal absorption of a secondary fuel is deflated by its own price index, so `use_change`
is `(X′/P̂) / X`: a value above one means the country burns more of that fuel. Multiply by an
emission factor to obtain a carbon account.

Columns: `country`, `sector`, `use`, `use_new`, `use_change`.
"""
function fossil_use(r::KiteResult{MahlkowWanner2023})
    b = r.baseline
    m = r.model
    country = String[]; sector = String[]
    use = Float64[]; use_new = Float64[]
    for s in m.secondary, d in 1:b.N
        push!(country, b.countries[d])
        push!(sector, b.sectors[s])
        push!(use, b.X[d, s])
        push!(use_new, r.X′[d, s] / r.P̂[d, s])
    end
    return DataFrame(country = country, sector = sector, use = use, use_new = use_new,
                     use_change = _safe_ratio.(use_new, use))
end
