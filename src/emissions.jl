# Carbon accounting for Mahlkow & Wanner (2023), their §3.5, equations (16)–(18).
#
# The KITE whitepaper reproduces the paper's equilibrium system (its §2.4, equations 23–29) but
# not this part, so the equations here come from the paper itself.
#
# Three ways of assigning the same global total to countries:
#
#   production  (16)  where the fuel is burnt
#   consumption (17)  where the goods whose production burnt it are finally absorbed
#   extraction  (18)  where the fuel came out of the ground
#
# All three sum to the same world figure, which is the sharpest test of the implementation and
# is asserted in the test suite.
#
# Baseline price levels are normalised to one, as everywhere in exact hat algebra: only P̂ is
# identified. A carbon intensity χ is therefore CO₂ per baseline *dollar* of fuel absorbed, and
# a counterfactual quantity is X′/P̂. When χ is calibrated from a satellite account this is
# exactly what the calibration returns, so the two conventions agree by construction. It does
# mean χ is country-specific unless you supply a single global figure, because it absorbs
# baseline price differences across countries.
#
# ---------------------------------------------------------------------------------------------
# INDEX CONVENTION — read this before checking any line here against the paper.
#
# The paper writes π_{ni} for the share of country *n*'s expenditure sourced from *i*: buyer
# first. KITE stores π[o, d, j] with the seller first — origin, destination, sector — because
# that makes a fixed sector a contiguous N×N matrix. So
#
#     paper  π_{ni}^j   ==   code  π[i, n, j]          (the indices swap)
#
# Every transposition below follows from that one line, and it is the easiest thing in this file
# to get wrong. Equation (18) in particular chains two of them: the burner's sourcing of the
# fuel, and then the refiner's sourcing of the crude that made it.
#
# The aggregate identity Σ_n E_n = Σ_n CF_n = Σ_n EF_n is what catches a mistake — each sum
# telescopes only if every share is contracted over the right index. The test suite asserts it,
# and it did in fact catch a transposed multiplier during development.
# ---------------------------------------------------------------------------------------------

# ── the pieces both the baseline and the counterfactual need ──────────────────────────────

"""
    _fuel_absorption(r) -> (A, A′)

`(N, |S|)` real absorption of each burnt fuel, net of the part transformed by a Leontief
secondary sector and deflated by the fuel's own price index. This is the bracket of equation
(16); multiplying by `χ` gives emissions.
"""
function _fuel_absorption(r::KiteResult{MahlkowWanner2023})
    b, m = r.baseline, r.model
    corr, corr′ = _transformed_fuel(r)
    A = Matrix{Float64}(undef, b.N, length(m.burnt))
    A′ = similar(A)
    for (i, s) in enumerate(m.burnt), d in 1:b.N
        A[d, i] = b.X[d, s] - corr[d, i]
        A′[d, i] = (r.X′[d, s] - corr′[d, i]) / r.P̂[d, s]
    end
    return A, A′
end

"""
    _direct_intensity(r, is′) -> (q, q′)

`(N, J)` direct emissions per unit of gross output — what sector `k` in country `d` emits by
burning fuel bought as an intermediate,

    q[d,k] = Σ_{s ∈ S} χ[d,s] · input_share[d,s,k] / P^s_d,

with the purchase excluded where `k` is the Leontief sector that transforms `s` rather than
burning it. `is′` is the counterfactual input-share array from
[`_counterfactual_input_share`](@ref).

Summing `q ⊙ Y` over sectors and adding direct final-consumption burning reproduces equation
(16) exactly, which is asserted in the tests.
"""
function _direct_intensity(r::KiteResult{MahlkowWanner2023}, is′)
    b, m = r.baseline, r.model
    q = zeros(b.N, b.J)
    q′ = zeros(b.N, b.J)
    # (fuel, sector) pairs that are transformations rather than combustion
    transformed = Set{Tuple{Int,Int}}()
    for (i, t) in enumerate(m.leontief)
        push!(transformed, (m.complement[i], t))
    end
    for s in m.burnt, k in 1:b.J
        (s, k) in transformed && continue
        for d in 1:b.N
            χ = m.χ[d, s]
            χ == 0 && continue
            q[d, k] += χ * b.input_share[d, s, k]
            q′[d, k] += χ * is′[d, s, k] / r.P̂[d, s]
        end
    end
    return q, q′
end

"""
    _final_burning(r) -> (f, f′)

`(N)` emissions from fuel burnt in final consumption rather than by a sector — car fuel and
domestic heating. `Σ_{s ∈ S} χ[d,s] · α[d,s] · I_d / P^s_d`.
"""
function _final_burning(r::KiteResult{MahlkowWanner2023})
    b, m = r.baseline, r.model
    f = zeros(b.N)
    f′ = zeros(b.N)
    for s in m.burnt, d in 1:b.N
        χ = m.χ[d, s]
        χ == 0 && continue
        f[d] += χ * b.α[d, s] * b.I[d]
        f′[d] += χ * b.α[d, s] * r.I′[d] / r.P̂[d, s]
    end
    return f, f′
end

# ── (16) production footprint ─────────────────────────────────────────────────────────────

"""
    _production_footprint(r) -> (E, E′)

Mahlkow & Wanner equation (16): territorial emissions, `Σ_s χ^s (X^s_n − transformed) / P^s_n`.
Note this includes fuel burnt in final consumption, so it is a production footprint only in the
sense of "emitted on this territory".
"""
function _production_footprint(r::KiteResult{MahlkowWanner2023})
    m = r.model
    A, A′ = _fuel_absorption(r)
    E = zeros(r.baseline.N)
    E′ = zeros(r.baseline.N)
    for (i, s) in enumerate(m.burnt), d in 1:r.baseline.N
        E[d] += m.χ[d, s] * A[d, i]
        E′[d] += m.χ[d, s] * A′[d, i]
    end
    return E, E′
end

# ── (17) consumption footprint ────────────────────────────────────────────────────────────

"""
    _emission_multiplier(q, π, τ, ζ, input_share; tolerance, max_iterations) -> v

Total emissions released anywhere in the world per unit of final demand for good `(o, j)` —
the row vector `v′ = q′ (I − A)⁻¹` of equation (17), where `A` is the global input-coefficient
matrix `A[(o,j),(d,k)] = π[o,d,j] · input_share[d,j,k] / (τ ζ)`.

The inverse is never formed. At 196 × 133 it would be a 26 068² dense factorisation — 5.4 GB
and hours — whereas `v` solves the single transposed system `(I − A)′ v = q`, i.e. `v = q + A′v`,
which reads

    h[o,k] = Σ_d π[d,o,k] · v[d,k] / (τ[d,o,k] ζ[d,o,k])      emissions embodied in a unit of
                                                              good k as bought by country o
    v[o,j] = q[o,j] + Σ_k input_share[o,k,j] · h[o,k]         direct, plus everything embodied
                                                              in the inputs o buys to make j

It converges at the spectral radius of the input-output matrix — the same rate as the solver's
own expenditure loop — and each pass is one `O(N²J)` and one `O(NJ²)` contraction, both over the
contiguous first dimension.

Tariff and export-subsidy wedges divide rather than multiply here on purpose: revenue collected
at the border is not a payment to any producer, so it leaves the production chain and reappears
as income. This is the same `π/(τζ)` convention the goods-market identity uses, which is what
makes `Σ_n CF_n` come out equal to `Σ_n E_n` exactly.
"""
function _emission_multiplier(q::AbstractMatrix, π::AbstractArray{<:Real,3},
                              τ::AbstractArray{<:Real,3}, ζ::AbstractArray{<:Real,3},
                              input_share::AbstractArray{<:Real,3};
                              tolerance::Float64 = 1e-13, max_iterations::Int = 100_000)
    N, J = size(q)
    v = copy(q)
    v_prev = similar(v)
    h = Matrix{Float64}(undef, N, J)
    for _ in 1:max_iterations
        copyto!(v_prev, v)
        @inbounds for k in 1:J, o in 1:N
            s = 0.0
            @simd for d in 1:N
                s += π[d, o, k] * v[d, k] / (τ[d, o, k] * ζ[d, o, k])
            end
            h[o, k] = s
        end
        copyto!(v, q)
        @inbounds for j in 1:J, k in 1:J
            @views @. v[:, j] += input_share[:, k, j] * h[:, k]
        end
        crit = 0.0
        @inbounds for i in eachindex(v)
            crit = max(crit, abs(v[i] - v_prev[i]) / max(abs(v[i]), 1e-12))
        end
        crit < tolerance && return v
    end
    @warn "emission multiplier did not converge; the input-output system may be near-singular."
    return v
end

"""
    _consumption_footprint_sector(r, is′) -> (CF, CF′)

`(N, J)` — the emissions released worldwide to serve each country's final demand **for each
sector's goods**: the inner term of equation (17), before it is summed over sectors.

    CF[n,j] = Σ_o v[o,j] · π[o,n,j] · α[n,j] I_n / (τ ζ)

**Indexed by the sector of the final good, not by the sector that emitted.** The steel in a car
lands on vehicles, not on iron and steel. That is exactly what makes the column sum meaningful —
add the direct final burning of [`_final_burning`](@ref) and it is the country's consumption
footprint, to machine precision — and it is also the thing that has to be said out loud wherever
the number is displayed, because the natural reading of "emissions by sector" is the other one.
"""
function _consumption_footprint_sector(r::KiteResult{MahlkowWanner2023}, is′)
    b, sc = r.baseline, r.scenario
    q, q′ = _direct_intensity(r, is′)

    v = _emission_multiplier(q, b.π, b.τ, b.ζ, b.input_share)
    v′ = _emission_multiplier(q′, r.π′, sc.τ′, sc.ζ′, is′)

    CF = zeros(b.N, b.J)
    CF′ = zeros(b.N, b.J)
    @inbounds for j in 1:b.J, n in 1:b.N
        base = b.α[n, j] * b.I[n]        # n's final demand for sector j, in purchaser prices
        new = b.α[n, j] * r.I′[n]
        s1 = 0.0; s2 = 0.0
        for o in 1:b.N
            # π/(τζ) converts a dollar of n's spending into the dollar the producer o receives,
            # which is the unit v is per. The wedge drops out of the chain because it is revenue,
            # not a payment for goods, and reappears in income.
            s1 += v[o, j] * b.π[o, n, j] / (b.τ[o, n, j] * b.ζ[o, n, j])
            s2 += v′[o, j] * r.π′[o, n, j] / (sc.τ′[o, n, j] * sc.ζ′[o, n, j])
        end
        CF[n, j] = s1 * base
        CF′[n, j] = s2 * new
    end
    return CF, CF′
end

"""
    _consumption_footprint(r, is′) -> (CF, CF′)

Mahlkow & Wanner equation (17): the emissions released worldwide to serve each country's final
demand, plus the fuel that country burns directly in final consumption.

The sectoral term is summed here rather than duplicated, so the country and sector levels cannot
drift apart. Fuel a household burns itself never entered any production chain, so no multiplier
applies to it and it is simply added.
"""
function _consumption_footprint(r::KiteResult{MahlkowWanner2023}, is′)
    fin, fin′ = _final_burning(r)
    CF, CF′ = _consumption_footprint_sector(r, is′)
    return vec(sum(CF, dims = 2)) .+ fin, vec(sum(CF′, dims = 2)) .+ fin′
end

"""
    _counterfactual_input_share(r) -> Array{Float64,3}

`input_share` after the Leontief cost shares have moved: the complementary-fuel row of each
secondary sector is scaled by `γ̂_fuel` and the rest by `γ̂_rest`. Every other sector is
Cobb-Douglas, so its shares are unchanged.
"""
function _counterfactual_input_share(r::KiteResult{MahlkowWanner2023})
    b, m, st = r.baseline, r.model, r.ext
    isempty(m.leontief) && return b.input_share
    out = copy(b.input_share)
    @inbounds for (i, t) in enumerate(m.leontief)
        qc = m.complement[i]
        for k in 1:b.J
            g = k == qc ? view(st.fuel_cost_share_change, :, i) :
                          view(st.other_cost_share_change, :, i)
            @views @. out[:, k, t] *= g
        end
    end
    return out
end

# ── (18) extraction footprint ─────────────────────────────────────────────────────────────

"""
    _extraction_footprint(r) -> (EF, EF′)

Mahlkow & Wanner equation (18): emissions attributed to whoever pulled the carbon out of the
ground, following Kortum & Weisbach.

Two channels. A fuel burnt as extracted is traced by the burner's sourcing shares for that fuel.
A fuel that was refined or distributed first is traced through the *secondary producer's*
sourcing of the complementary primary fuel — the emissions from Dutch-refined petroleum belong
to whoever sold the Netherlands its crude, not to the Netherlands.
"""
function _extraction_footprint(r::KiteResult{MahlkowWanner2023})
    b, m = r.baseline, r.model
    A, A′ = _fuel_absorption(r)
    EF = zeros(b.N)
    EF′ = zeros(b.N)
    leo_of = Dict(t => i for (i, t) in enumerate(m.leontief))

    for (i, s) in enumerate(m.burnt)
        if haskey(leo_of, s)
            # ---- fuel that was processed first: two hops back to the ground ----------------
            # Paper: Σ_i π_{in}^{p^s} · ι^s · Σ_m π_{mi}^s X_m^s / P_m^s. Reading the inner sum
            # first, it is the emissions from fuel s *produced by* i; the outer share then asks
            # where i bought the crude. Emissions from Dutch-refined petroleum belong to whoever
            # sold the Netherlands its oil, not to the Netherlands.
            qc = m.complement[leo_of[s]]
            for o in 1:b.N                      # o refined or distributed the fuel
                e = 0.0; e′ = 0.0
                for dd in 1:b.N                 # dd bought it from o and burnt it
                    # π[o, dd, s] is dd's share sourced from o — the paper's π_{dd,o}^s
                    e += b.π[o, dd, s] * m.χ[dd, s] * A[dd, i]
                    e′ += r.π′[o, dd, s] * m.χ[dd, s] * A′[dd, i]
                end
                (e == 0 && e′ == 0) && continue
                for n in 1:b.N                  # n extracted the primary fuel o refined
                    # π[n, o, qc] is o's share of the primary fuel sourced from n
                    EF[n] += b.π[n, o, qc] * e
                    EF′[n] += r.π′[n, o, qc] * e′
                end
            end
        else
            # ---- fuel burnt as extracted: one hop ------------------------------------------
            # Paper: Σ_i π_{in}^p (X_i^p − transformed)/P_i^p. Coal is bought and burnt as it
            # comes, so the burner's own sourcing shares say who dug it up.
            for dd in 1:b.N                     # dd burnt it
                e = m.χ[dd, s] * A[dd, i]
                e′ = m.χ[dd, s] * A′[dd, i]
                (e == 0 && e′ == 0) && continue
                for n in 1:b.N                  # n extracted it
                    EF[n] += b.π[n, dd, s] * e
                    EF′[n] += r.π′[n, dd, s] * e′
                end
            end
        end
    end
    # Σ_n EF_n telescopes to global emissions because every sourcing share sums to one over its
    # origin index — which is exactly what the aggregate identity in the tests checks.
    return EF, EF′
end

# ── public interface ──────────────────────────────────────────────────────────────────────

"""
    emissions(r::KiteResult{MahlkowWanner2023}; level = :country) -> DataFrame

Carbon accounts for a solved counterfactual, after Mahlkow & Wanner (2023) §3.5.

Requires the model to carry an `emission_intensity`; see [`MahlkowWanner2023`](@ref) and
[`emission_intensity_from_satellite`](@ref).

`level`:

- `:country` — the three footprints of equations (16)–(18), baseline and counterfactual, with
  the ratio. Each of the three sums to the same world total, before and after.
- `:fuel` — production emissions split by burnt fuel, `(country, sector)`.
- `:sector` — **two** of the three footprints, per `(country, sector)`, plus a `households` row
  per country for fuel burnt in final consumption:

  * `production` — emissions by the sector that *burns* the fuel, `q ⊙ Y`. The layout of a
    standard MRIO satellite account, so it is what you compare against the data.
  * `consumption` — emissions released worldwide to serve that country's final demand for that
    sector's goods, indexed by the sector of the **final good** rather than by the sector that
    emitted: the steel in a car lands on vehicles, not on iron and steel. The two rankings
    genuinely differ — on EMERGING 2023 construction is the largest consumption figure in the
    world while barely registering as a burner.

  Each column sums, over the sectors of one country, to that country's footprint at
  `level = :country`. There is **no sectoral extraction footprint**: extraction is attributed to
  whoever took the fuel out of the ground, which happens only in the `primary` sectors, so the
  sectoral cut of it is `level = :fuel`.

  Note this level costs two [`_emission_multiplier`](@ref) solves, as `:country` does.

Units are those of the supplied intensity: if `χ` is tonnes of CO₂ per baseline dollar, these
are tonnes.

# Examples
```julia
e = emissions(r)
sum(e.production_new) - sum(e.consumption_new)     # ≈ 0, the accounting identity
```
"""
function emissions(r::KiteResult{MahlkowWanner2023}; level::Symbol = :country)
    m = r.model
    m.has_carbon || error("emissions: this model carries no emission_intensity, so there is " *
                          "nothing to account. Pass `emission_intensity` to " *
                          "MahlkowWanner2023, e.g. from emission_intensity_from_satellite.")
    level in (:country, :fuel, :sector) ||
        error("level must be :country, :fuel or :sector; got :$level.")
    b = r.baseline

    if level === :country
        E, E′ = _production_footprint(r)
        CF, CF′ = _consumption_footprint(r, _counterfactual_input_share(r))
        EF, EF′ = _extraction_footprint(r)
        return DataFrame(country = b.countries,
                         production = E, production_new = E′,
                         production_change = _safe_ratio.(E′, E),
                         consumption = CF, consumption_new = CF′,
                         consumption_change = _safe_ratio.(CF′, CF),
                         extraction = EF, extraction_new = EF′,
                         extraction_change = _safe_ratio.(EF′, EF))
    elseif level === :fuel
        A, A′ = _fuel_absorption(r)
        country = String[]; sector = String[]; val = Float64[]; val′ = Float64[]
        for (i, s) in enumerate(m.burnt), d in 1:b.N
            push!(country, b.countries[d]); push!(sector, b.sectors[s])
            push!(val, m.χ[d, s] * A[d, i]); push!(val′, m.χ[d, s] * A′[d, i])
        end
        return DataFrame(country = country, sector = sector,
                         production = val, production_new = val′,
                         production_change = _safe_ratio.(val′, val))
    else
        is′ = _counterfactual_input_share(r)
        q, q′ = _direct_intensity(r, is′)
        fin, fin′ = _final_burning(r)
        CF, CF′ = _consumption_footprint_sector(r, is′)
        country = String[]; sector = String[]
        val = Float64[]; val′ = Float64[]
        cf = Float64[]; cf′ = Float64[]
        for k in 1:b.J, d in 1:b.N
            push!(country, b.countries[d]); push!(sector, b.sectors[k])
            push!(val, q[d, k] * b.Y[d, k]); push!(val′, q′[d, k] * r.Y′[d, k])
            push!(cf, CF[d, k]); push!(cf′, CF′[d, k])
        end
        for d in 1:b.N
            push!(country, b.countries[d]); push!(sector, "households")
            push!(val, fin[d]); push!(val′, fin′[d])
            # fuel a household burns itself is emitted and consumed by the same household, so
            # the two columns coincide on this row — and including it is what makes each column
            # sum to its country-level footprint rather than to almost it
            push!(cf, fin[d]); push!(cf′, fin′[d])
        end
        return DataFrame(country = country, sector = sector,
                         production = val, production_new = val′,
                         production_change = _safe_ratio.(val′, val),
                         consumption = cf, consumption_new = cf′,
                         consumption_change = _safe_ratio.(cf′, cf))
    end
end

# ── calibrating χ from a satellite account ────────────────────────────────────────────────
#
# Two routes, because the two databases document their emissions differently.
#
#   * EMERGING ships CO₂ by (country, sector, fuel type). Aggregating over sectors gives CO₂ by
#     country and fuel, which divides straight into fuel absorption — exact, no estimation.
#   * OECD's TeCO2 and most other MRIO satellites give CO₂ by (country, industry) only. The
#     fuel split then has to be inferred from which fuels each industry buys, which is a small
#     non-negative least-squares problem per country.
#
# Both return CO₂ per baseline dollar of fuel absorbed, net of transformed fuel, which is the
# convention `emissions` expects.

"""
    _absorption_matrices(b, m) -> (U, A)

`U[d,s,k]` is country `d`'s sector `k` purchase of burnt fuel `s`, excluding purchases that a
Leontief sector transforms rather than burns; `A[d,i]` is total net absorption of the `i`-th
burnt fuel, intermediates plus final consumption. These are the baseline quantities every
calibration divides into.
"""
function _absorption_matrices(b::KiteBaseline, m::MahlkowWanner2023)
    ns = length(m.burnt)
    U = zeros(b.N, ns, b.J)
    A = zeros(b.N, ns)
    transformed = Set{Tuple{Int,Int}}()
    for (i, t) in enumerate(m.leontief)
        push!(transformed, (m.complement[i], t))
    end
    for (i, s) in enumerate(m.burnt)
        for k in 1:b.J
            (s, k) in transformed && continue
            @views @. U[:, i, k] = b.input_share[:, s, k] * b.Y[:, k]
        end
        @views @. A[:, i] = b.α[:, s] * b.I                 # final consumption
        for k in 1:b.J
            @views @. A[:, i] += U[:, i, k]
        end
    end
    return U, A
end

"""
    emission_intensity_from_fuel_co2(b, model, co2; verbose = 1) -> Matrix{Float64}

Carbon intensity from a satellite account that reports CO₂ **by fuel**, the EMERGING layout.

`co2` is `(N, |S|)`, country by burnt fuel, in the `secondary` order of `model`. The intensity
is CO₂ divided by that fuel's net absorption, so baseline production footprints reproduce the
satellite exactly, country by country and fuel by fuel.

Countries that absorb none of a fuel get zero, and are reported when `verbose > 0`.

# Examples
```julia
χ = emission_intensity_from_fuel_co2(b, model, co2)     # co2[:, i] matches model.burnt_codes[i]
model = MahlkowWanner2023(b; primary, secondary, emission_intensity = χ)
```
"""
function emission_intensity_from_fuel_co2(b::KiteBaseline, m::MahlkowWanner2023,
                                          co2::AbstractMatrix; verbose::Int = 1)
    ns = length(m.burnt)
    size(co2) == (b.N, ns) ||
        error("co2 must be $(b.N)×$ns (country × burnt fuel, in `secondary` order); " *
              "got $(size(co2)).")
    all(≥(0), co2) || error("co2 must be non-negative.")
    _, A = _absorption_matrices(b, m)

    χ = zeros(b.N, b.J)
    missing_abs = 0
    for (i, s) in enumerate(m.burnt), d in 1:b.N
        if A[d, i] > 0
            χ[d, s] = co2[d, i] / A[d, i]
        elseif co2[d, i] > 0
            missing_abs += 1
        end
    end
    if verbose > 0
        @info @sprintf("emission_intensity_from_fuel_co2: %d fuels × %d countries, world CO₂ %.4g, mean intensity %.4g%s",
                       ns, b.N, sum(co2), sum(co2) / sum(A),
                       missing_abs == 0 ? "" :
                       "; $missing_abs cells report CO₂ with no matching absorption and are dropped")
    end
    return χ
end

"""
    emission_intensity_from_satellite(b, model, co2; iterations = 200, verbose = 1)
        -> Matrix{Float64}

Carbon intensity from a satellite account that reports CO₂ **by industry** rather than by fuel —
the OECD TeCO2 layout, and what most MRIO satellites publish.

`co2` is `(N, J)`, country by *burning* industry. Since the model needs one intensity per fuel,
the fuel split is inferred from which fuels each industry buys: for every country solve

    min_{χ ≥ 0}  Σ_k ( Σ_s χ_s · U[d,s,k] − co2[d,k] )²

by non-negative multiplicative updates. The industry pattern therefore sets the **shape** of the
fuel split; the **level** is then set by rescaling so that each country's baseline production
footprint reproduces the satellite's country total exactly. Splitting the two jobs this way is
what makes the baseline check meaningful: country totals are right by construction, and only the
allocation across fuels is estimated.

The rescaling treats the satellite as covering **all** combustion, including fuel burnt directly
in final consumption — car fuel and domestic heating. That is the right reading for a satellite
with no household row, such as EMERGING's, where private motoring sits inside a transport
industry. If yours genuinely excludes households, the level will be low by that amount.

`verbose > 0` reports the fit quality: the share of cross-industry variation the fuel split
explains. Treat a low figure as a warning about the **split**, not the totals.

**The split is only as well identified as industries differ in their fuel mix**, and only as
good as the satellite's industry attribution matches who buys the fuel in the input-output
table — emissions from private cars booked to a transport industry are a known mismatch, since
the fuel was bought by households. Prefer [`emission_intensity_from_fuel_co2`](@ref) whenever
the satellite reports fuels directly, as EMERGING's underlying file does: it needs no estimation
at all.

# Examples
```julia
χ = emission_intensity_from_satellite(b, model, co2)    # co2[:, k] is industry k's CO₂
```
"""
function emission_intensity_from_satellite(b::KiteBaseline, m::MahlkowWanner2023,
                                           co2::AbstractMatrix; iterations::Int = 200,
                                           verbose::Int = 1)
    size(co2) == (b.N, b.J) ||
        error("co2 must be $(b.N)×$(b.J) (country × burning industry); got $(size(co2)).")
    all(≥(0), co2) || error("co2 must be non-negative.")
    ns = length(m.burnt)
    U, A = _absorption_matrices(b, m)

    χ = zeros(b.N, b.J)
    ss_res = 0.0; ss_tot = 0.0
    final_share = 0.0
    for d in 1:b.N
        target = @view co2[d, :]
        Ud = @view U[d, :, :]                       # (ns, J)
        tot = sum(target)
        (tot > 0 && sum(A[d, :]) > 0) || continue
        interm = vec(sum(Ud, dims = 2))             # intermediate absorption by fuel

        # Fit x ≥ 0 minimising Σ_k (Σ_s x_s·U[s,k] − co2[k])² by multiplicative updates:
        #
        #     x_s  ←  x_s · (Σ_k U[s,k]·target[k]) / (Σ_k U[s,k]·predicted[k])
        #
        # The update is a ratio of two non-negative sums, so x can never go negative and the
        # constraint needs no projection, no active set and no external solver — which matters
        # because the alternative is a dependency for a problem with four unknowns. At the fixed
        # point the two sums are equal, which is the first-order condition. The problem is tiny
        # (|S| unknowns against J industries) and convergence is quick, but it is a local method
        # on a weakly identified problem, hence the identification caveat in the docstring.
        x = fill(tot / max(sum(interm), eps()), ns)
        pred = Vector{Float64}(undef, b.J)
        num = Vector{Float64}(undef, ns)
        den = Vector{Float64}(undef, ns)
        for _ in 1:iterations
            mul!(pred, transpose(Ud), x)                # predicted emissions by industry
            fill!(num, 0.0); fill!(den, 0.0)
            @inbounds for k in 1:b.J, i in 1:ns
                u = Ud[i, k]
                u == 0 && continue
                num[i] += u * target[k]
                den[i] += u * pred[k]
            end
            converged = true
            @inbounds for i in 1:ns
                den[i] <= 0 && continue                 # this fuel is bought by no industry
                f = num[i] / den[i]
                abs(f - 1) > 1e-10 && (converged = false)
                x[i] *= f
            end
            converged && break
        end

        # Set the level so the baseline production footprint reproduces the satellite total.
        # Rescaling on intermediate absorption alone would be a trap: a fuel sold mostly to
        # households — distributed gas — has little intermediate use, so its intensity would be
        # inflated by the ratio of total to intermediate absorption and the world footprint
        # would come out orders of magnitude too high.
        implied = dot(A[d, :], x)
        implied > 0 && (x .*= tot / implied)
        final_share += dot(A[d, :] .- interm, x)

        mul!(pred, transpose(Ud), x)
        m̄ = sum(target) / b.J
        @inbounds for k in 1:b.J
            ss_res += (pred[k] - target[k])^2
            ss_tot += (target[k] - m̄)^2
        end
        for (i, s) in enumerate(m.burnt); χ[d, s] = x[i]; end
    end
    if verbose > 0
        r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
        @info @sprintf("emission_intensity_from_satellite: %d fuels fitted to %d industries; country totals reproduce the satellite (%.4g) exactly, of which %.1f%% is fuel burnt in final consumption; the fuel split explains %.1f%% of cross-industry variation",
                       ns, b.J, sum(co2), 100 * final_share / sum(co2), 100 * r2)
    end
    return χ
end

# ── carbon pricing ────────────────────────────────────────────────────────────────────────

"""
    set_carbon_price!(sc, b, model, price; country = :all, sector = :all,
                      basis = :specific, mode = :set)

Price the carbon released when fossil fuel is burnt.

A carbon price falls on a country's **absorption** of fuel, wherever the fuel came from, so it
is an origin-neutral wedge — unlike a tariff, it applies to domestic supply too. That is exactly
a `τ′` set uniformly across origins *including the diagonal*, which is how it is implemented:
the existing machinery then collects the revenue on the whole domestic base and hands it back as
income, and leaves sourcing shares undistorted because the wedge is common to every origin.

`price` is money per unit of CO₂, in the units the model's emission intensity uses. With the
EMERGING convention — value in USD million, emissions in Mt CO₂ — the intensity is numerically
tonnes of CO₂ per dollar, so `price` is simply **USD per tonne of CO₂**.

`basis`:

- `:specific` (default) — a genuine price per tonne. Its ad-valorem equivalent is
  `1 + price·χ/P̂ᵖʳᵉ`, where `P̂ᵖʳᵉ` is the fuel price *net of the tax itself*, so the solver
  revises the wedge as it iterates. This is what real carbon pricing is, and the distinction is
  not second-order: at 100 USD/t a coal intensity of 0.021 t/USD is a 210% ad-valorem wedge, and
  the fuel price moves a long way under it.
- `:ad_valorem` — freeze the wedge at its baseline-price value, `1 + price·χ`. Cheaper, works
  with any model rather than only [`MahlkowWanner2023`](@ref), and accurate for small prices.

`mode` is `:set` or `:add`, the latter stacking on an existing carbon price.

A border carbon adjustment is this plus an ordinary [`set_tariff!`](@ref) on the carbon-intensive
imports: the domestic price is what the adjustment exists to protect, so both halves are needed
for the counterfactual to be coherent. The two compose — the carbon wedge multiplies onto
whatever `τ′` already holds — so the order they are applied in does not matter.

# One limit worth knowing

The wedge is indexed by `(origin, destination, fuel)`. It cannot tell one *buyer* apart from
another within the destination, so where a burnt fuel is also the complementary input of a
Leontief secondary sector — natural gas bought by gas distribution, `"GASD" => "GAS"` — the
part that is transformed rather than burnt is taxed here **and** again when the distributed
product is absorbed. That chain therefore carries more than `price` per tonne.

Real schemes exempt the transformation input for exactly this reason, and no destination-uniform
wedge can reproduce that. It affects only fuels appearing both in `secondary` and as another
entry's complement — on the EMERGING taxonomy that is natural gas alone, coal and refined
petroleum feeding no further transformation. Scaling the wedge down by the combusted share was
considered and rejected: it would under-price the directly-burnt majority to compensate for the
transformed minority, trading a visible bias for a hidden one.

# Examples
```julia
sc = Scenario(b; label = "EU ETS at 100 USD/t")
set_carbon_price!(sc, b, model, 100.0; country = eu_members)
r = update_equilibrium(model, b, sc; vfactor = 0.05)
emissions(r)
```

See also [`MahlkowWanner2023`](@ref), [`emissions`](@ref).
"""
function set_carbon_price!(sc::Scenario, b, model::MahlkowWanner2023, price::Real;
                           country = :all, sector = :all, basis::Symbol = :specific,
                           mode::Symbol = :set)
    basis in (:specific, :ad_valorem) ||
        error("basis must be :specific or :ad_valorem; got :$basis.")
    mode in (:set, :add) || error("mode must be :set or :add; got :$mode.")
    model.has_carbon ||
        error("set_carbon_price!: the model carries no emission_intensity, so there is no " *
              "carbon content to price. Pass `emission_intensity` to MahlkowWanner2023.")
    price ≥ 0 || error("a carbon price must be non-negative; got $price.")

    ds = _countries(b, country)
    js = intersect(_sectors(b, sector), model.burnt)
    isempty(js) && @warn "set_carbon_price!: no burnt fuel matches `sector`, so nothing is priced."

    for j in js, d in ds
        w = price * model.χ[d, j]
        sc.carbon[d, j] = mode === :set ? w : sc.carbon[d, j] + w
    end
    if basis === :specific
        sc.carbon_specific = true
    else
        # Freeze at baseline prices: evaluate the wedge once, here, and never again. It still
        # goes through `_apply_carbon_wedge!` so that it composes with any tariff on the same
        # cell and so that re-setting the price replaces rather than compounds it.
        _apply_carbon_wedge!(sc, ones(size(sc.carbon)); inclusive = false)
    end
    return sc
end

