# The shared outer loop. The two models differ only in `_wage_update!` and `_transfers!`.

"""
    update_equilibrium(model, baseline, scenario = Scenario(baseline); kwargs...) -> KiteResult
    update_equilibrium(model, baseline, scenario, settings::SolverSettings) -> KiteResult

Solve for the counterfactual equilibrium in changes ("exact hat algebra").

`model` is [`CaliendoParro2015`](@ref) or [`ChowdhryHinzKaminWanner2022`](@ref); `baseline` is a
model-consistent [`KiteBaseline`](@ref); `scenario` holds the policy shock. Keyword arguments
are forwarded to [`SolverSettings`](@ref).

The solver iterates on the wage change `ŵ`. Each outer iteration updates input costs, sectoral
price indices and trade shares, then solves the expenditure/output block to convergence, and
finally revises `ŵ`. Because the baseline satisfies the equilibrium identities exactly, a
no-change scenario converges in a single iteration with every hat equal to one.

# Examples
```julia
r = update_equilibrium(CaliendoParro2015(), b)                    # no-change benchmark
r = update_equilibrium(CaliendoParro2015(), b, sc; tolerance = 1e-8)
r = update_equilibrium(ChowdhryHinzKaminWanner2022(), b, sc; vfactor = 0.1)
```
"""
function update_equilibrium(model::KiteModel, b::KiteBaseline,
                            sc::Scenario = Scenario(b); kwargs...)
    return update_equilibrium(model, b, sc, SolverSettings(; kwargs...))
end

function update_equilibrium(model::KiteModel, b::KiteBaseline, sc::Scenario,
                            settings::SolverSettings)
    t0 = time()
    _check_scenario(model, b, sc)
    _check_conformable(b, sc)

    ws = _Workspace(b, sc)
    num_index = _numeraire_index(b, settings.numeraire)

    criterion = Inf
    inner_ok = true
    inner_total = 0
    iter = 0

    while iter < settings.max_iterations
        iter += 1
        copyto!(ws.ŵ_prev, ws.ŵ)

        _input_cost!(ws, b, sc)
        _price_index!(ws, b)
        _trade_shares!(ws, b)
        _policy_weights!(ws, b, sc)

        n_inner, inner_ok = _expenditure_block!(ws, b, sc, settings)
        inner_total += n_inner

        _value_added!(ws, b)
        _update_trade_balance!(ws, b, settings.trade_balance_rule)
        _transfers!(model, ws, b, sc)
        _wage_update!(model, ws, b, sc, settings)
        _normalise_wage!(ws, b, num_index)

        criterion = _criterion(ws.ŵ, ws.ŵ_prev, settings.convergence)

        if settings.verbose ≥ 2
            @printf("  iteration %4d   criterion %.3e   inner %d\n", iter, criterion, n_inner)
        end

        converged = criterion ≤ settings.tolerance &&
                    (!settings.require_inner_convergence || inner_ok)
        converged && break
    end

    converged = isfinite(criterion) && criterion ≤ settings.tolerance &&
                (!settings.require_inner_convergence || inner_ok)
    elapsed = time() - t0

    if settings.verbose ≥ 1
        if converged
            @info @sprintf("%s: converged in %d iterations (criterion %.2e, %.3g s)",
                           _model_name(model), iter, criterion, elapsed)
        else
            @warn @sprintf("%s: did NOT converge in %d iterations (criterion %.2e). \
                            Try a smaller vfactor or more iterations.",
                           _model_name(model), iter, criterion)
        end
    end

    return KiteResult(model, b, sc, settings,
                      copy(ws.ŵ), copy(ws.ĉ), copy(ws.P̂), copy(ws.π′),
                      copy(ws.X′), copy(ws.Y′), copy(ws.I′), copy(ws.VA′),
                      copy(ws.D′), copy(ws.T′),
                      converged, criterion, iter, inner_total, elapsed)
end

"""
    caliendo_parro_2015(baseline, scenario = Scenario(baseline); kwargs...) -> KiteResult

Convenience wrapper for [`update_equilibrium`](@ref) with [`CaliendoParro2015`](@ref).
"""
caliendo_parro_2015(b::KiteBaseline, sc::Scenario = Scenario(b); kwargs...) =
    update_equilibrium(CaliendoParro2015(), b, sc; kwargs...)

"""
    chowdhry_hinz_kamin_wanner_2022(baseline, scenario = Scenario(baseline); kwargs...) -> KiteResult

Convenience wrapper for [`update_equilibrium`](@ref) with
[`ChowdhryHinzKaminWanner2022`](@ref).
"""
chowdhry_hinz_kamin_wanner_2022(b::KiteBaseline, sc::Scenario = Scenario(b); kwargs...) =
    update_equilibrium(ChowdhryHinzKaminWanner2022(), b, sc; kwargs...)

# ── validation ────────────────────────────────────────────────────────────────────────────

function _check_conformable(b::KiteBaseline, sc::Scenario)
    _check_size(sc.τ′, (b.N, b.N, b.J), "scenario τ′")
    _check_size(sc.ζ′, (b.N, b.N, b.J), "scenario ζ′")
    _check_size(sc.κ̂, (b.N, b.N, b.J), "scenario κ̂")
    _check_size(sc.ẑ, (b.N, b.J), "scenario ẑ")
    _check_size(sc.L̂, (b.N,), "scenario L̂")
    _check_size(sc.coalition, (b.N,), "scenario coalition")
    all(>(0), sc.τ′) || error("scenario τ′ must be strictly positive.")
    all(>(0), sc.ζ′) || error("scenario ζ′ must be strictly positive.")
    all(>(0), sc.κ̂) || error("scenario κ̂ must be strictly positive.")
    all(>(0), sc.ẑ) || error("scenario ẑ must be strictly positive.")
    all(>(0), sc.L̂) || error("scenario L̂ must be strictly positive.")
    return nothing
end

function _numeraire_index(b::KiteBaseline, numeraire)
    numeraire isa Symbol && return 0
    haskey(b.country_index, numeraire) ||
        error("numeraire country \"$numeraire\" is not in the baseline.")
    return b.country_index[numeraire]
end

# ── the equilibrium blocks ────────────────────────────────────────────────────────────────

"""
    _input_cost!(ws, b, sc)

`ĉ[d,j] = (1/ẑ[d,j]) · exp( β[d,j]·log ŵ[d] + (1 − β[d,j])·Σ_k γ[d,k,j]·log P̂[d,k] )`
"""
function _input_cost!(ws::_Workspace, b::KiteBaseline, sc::Scenario)
    N, J = ws.N, ws.J
    @inbounds @. ws.logP̂ = log(ws.P̂)
    @inbounds for j in 1:J
        cj = view(ws.ĉ, :, j)
        fill!(cj, 0.0)
        for k in 1:J
            @views @. cj += b.γ[:, k, j] * ws.logP̂[:, k]
        end
        @views @. cj = exp(b.β[:, j] * log(ws.ŵ) + (1 - b.β[:, j]) * cj)
    end
    ws.has_productivity && @inbounds @. ws.ĉ /= sc.ẑ
    return ws
end

"""
    _price_index!(ws, b)

`P̂[d,j] = ( Σ_o π[o,d,j] · (φ̂[o,d,j]·ĉ[o,j])^(-θ_j) )^(-1/θ_j)`, evaluated as one `gemv!` per
sector against the cached `W = π·φ̂^(-θ)`.
"""
function _price_index!(ws::_Workspace, b::KiteBaseline)
    J = ws.J
    @inbounds for j in 1:J
        θj = b.θ[j]
        @views @. ws.u = ws.ĉ[:, j]^(-θj)
        Wj = view(ws.W, :, :, j)
        mul!(ws.t, transpose(Wj), ws.u)
        @views @. ws.P̂[:, j] = ifelse(ws.t > 0, ws.t^(-1 / θj), 1.0)
    end
    return ws
end

"""
    _trade_shares!(ws, b)

`π′[o,d,j] = π[o,d,j]·(φ̂[o,d,j]·ĉ[o,j]/P̂[d,j])^(-θ_j) = W[o,d,j]·ĉ[o,j]^(-θ_j)·P̂[d,j]^(θ_j)`
"""
function _trade_shares!(ws::_Workspace, b::KiteBaseline)
    N, J = ws.N, ws.J
    @inbounds for j in 1:J
        θj = b.θ[j]
        @views @. ws.u = ws.ĉ[:, j]^(-θj)
        for d in 1:N
            vd = ws.P̂[d, j]^θj
            @views @. ws.π′[:, d, j] = ws.W[:, d, j] * ws.u * vd
        end
    end
    return ws
end

"""
    _policy_weights!(ws, b, sc)

Refresh the per-outer-iteration caches `A = π′/(τ′ζ′)`, `Aζ = (ζ′−1)·A` and
`tr_share[d,j] = Σ_o (τ′−1)/τ′·π′`.
"""
function _policy_weights!(ws::_Workspace, b::KiteBaseline, sc::Scenario)
    N, J = ws.N, ws.J
    @inbounds for j in 1:J, d in 1:N
        @views @. ws.A[:, d, j] = ws.π′[:, d, j] / (sc.τ′[:, d, j] * sc.ζ′[:, d, j])
    end
    if ws.has_export_subsidy
        @inbounds @. ws.Aζ = (sc.ζ′ - 1) * ws.A
    end
    if ws.has_tariff
        @inbounds for j in 1:J, d in 1:N
            acc = 0.0
            @simd for o in 1:N
                t = sc.τ′[o, d, j]
                acc += (t - 1) / t * ws.π′[o, d, j]
            end
            ws.tr_share[d, j] = acc
        end
    else
        fill!(ws.tr_share, 0.0)
    end
    return ws
end

"""
    _value_added!(ws, b)

`VA′[o] = Σ_j β[o,j]·Y′[o,j]`
"""
function _value_added!(ws::_Workspace, b::KiteBaseline)
    fill!(ws.VA′, 0.0)
    @inbounds for j in 1:ws.J
        @views @. ws.VA′ += b.β[:, j] * ws.Y′[:, j]
    end
    return ws
end

"""
    _update_trade_balance!(ws, b, rule::Symbol)

Counterfactual trade balance under `:fixed`, `:fixed_country_share` (constant relative to own
value added), `:fixed_global_share` (constant share of world value added) or `:zero`.
"""
function _update_trade_balance!(ws::_Workspace, b::KiteBaseline, rule::Symbol)
    if rule === :fixed
        copyto!(ws.D′, b.D)
    elseif rule === :zero
        fill!(ws.D′, 0.0)
    elseif rule === :fixed_country_share
        @inbounds @. ws.D′ = ifelse(b.VA == 0, 0.0, b.D / b.VA * ws.VA′)
    elseif rule === :fixed_global_share
        s = sum(b.VA)
        s_new = sum(ws.VA′)
        @inbounds @. ws.D′ = b.D / s * s_new
    else
        error("trade_balance_rule must be :fixed, :fixed_country_share, :fixed_global_share " *
              "or :zero; got :$rule.")
    end
    return ws
end

"""
    _normalise_wage!(ws, b, num_index)

Impose the numéraire: `num_index == 0` scales `ŵ` so world value added is unchanged, otherwise
it fixes the wage of country `num_index`. Real quantities are invariant to the choice.
"""
function _normalise_wage!(ws::_Workspace, b::KiteBaseline, num_index::Int)
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
    return ws
end

# ── expenditure / output block ────────────────────────────────────────────────────────────

function _expenditure_block!(ws::_Workspace, b::KiteBaseline, sc::Scenario,
                             settings::SolverSettings)
    if settings.inner_solver === :direct
        _direct_expenditure!(ws, b, sc)
        return 1, true
    end
    return _inner_expenditure!(ws, b, sc, settings)
end

"""
    _inner_expenditure!(ws, b, sc, settings) -> (iterations, converged)

Warm-started fixed point in `(X′, I′, Y′)`:

    X′[d,k] = α[d,k]·I′[d] + Σ_j input_share[d,k,j]·Y′[d,j]
    I′[d]   = Σ_j tr_share[d,j]·X′[d,j] + ES′[d] + L̂[d]·ŵ[d]·VA[d] − D′[d] + T′[d]
    Y′[o,j] = Σ_d A[o,d,j]·X′[d,j]

Convergence is measured on `Y′`. Because the workspace carries the previous outer iteration's
iterate, this typically needs a handful of passes — and exactly one when nothing has changed.
"""
function _inner_expenditure!(ws::_Workspace, b::KiteBaseline, sc::Scenario,
                             settings::SolverSettings)
    N, J = ws.N, ws.J
    converged = false
    it = 0

    while it < settings.max_inner_iterations
        it += 1
        copyto!(ws.Y_prev, ws.Y′)

        # expenditure from the previous income and output iterates
        _intermediate_demand!(ws.ID, b.input_share, ws.Y′)
        @inbounds for j in 1:J
            @views @. ws.X′[:, j] = b.α[:, j] * ws.I′ + ws.ID[:, j]
        end

        # export-subsidy cost accruing to the origin
        if ws.has_export_subsidy
            fill!(ws.ES′, 0.0)
            @inbounds for j in 1:J
                @views mul!(ws.t, view(ws.Aζ, :, :, j), ws.X′[:, j])
                @. ws.ES′ += ws.t
            end
        end

        # income
        @inbounds for d in 1:N
            acc = 0.0
            @simd for j in 1:J
                acc += ws.tr_share[d, j] * ws.X′[d, j]
            end
            lw = ws.has_population ? sc.L̂[d] * ws.ŵ[d] : ws.ŵ[d]
            ws.I′[d] = acc + (ws.has_export_subsidy ? ws.ES′[d] : 0.0) +
                       lw * b.VA[d] - ws.D′[d] + ws.T′[d]
        end

        # gross output
        @inbounds for j in 1:J
            @views mul!(ws.Y′[:, j], view(ws.A, :, :, j), ws.X′[:, j])
        end

        crit = _criterion(ws.Y′, ws.Y_prev, settings.convergence)
        isfinite(crit) || break
        if crit ≤ settings.inner_tolerance
            converged = true
            break
        end
    end
    return it, converged
end

"""
    _direct_expenditure!(ws, b, sc)

Solve the same block as a dense `NJ × NJ` linear system,

    (𝕀 − M) vec(X′) = vec(α ⊙ G),
    M[(d,k),(m,j)] = α[d,k]·(tr_share[d,j]·δ_{dm} + Aζ[d,m,j]) + input_share[d,k,j]·A[d,m,j],
    G[d] = L̂[d]·ŵ[d]·VA[d] − D′[d] + T′[d].

Exact up to round-off, and far more expensive than the iterative path — it exists as an
independent check that the fixed point really is the solution.
"""
function _direct_expenditure!(ws::_Workspace, b::KiteBaseline, sc::Scenario)
    N, J = ws.N, ws.J
    n = N * J
    M = zeros(n, n)
    idx(d, k) = d + N * (k - 1)

    @inbounds for j in 1:J, k in 1:J, m in 1:N, d in 1:N
        v = b.input_share[d, k, j] * ws.A[d, m, j]
        if d == m
            v += b.α[d, k] * ws.tr_share[d, j]
        end
        if ws.has_export_subsidy
            v += b.α[d, k] * ws.Aζ[d, m, j]
        end
        M[idx(d, k), idx(m, j)] += v
    end
    @inbounds for i in 1:n
        M[i, i] -= 1.0
    end

    rhs = Vector{Float64}(undef, n)
    @inbounds for k in 1:J, d in 1:N
        lw = ws.has_population ? sc.L̂[d] * ws.ŵ[d] : ws.ŵ[d]
        rhs[idx(d, k)] = -b.α[d, k] * (lw * b.VA[d] - ws.D′[d] + ws.T′[d])
    end

    x = M \ rhs
    copyto!(ws.X′, x)

    if ws.has_export_subsidy
        fill!(ws.ES′, 0.0)
        @inbounds for j in 1:J
            @views mul!(ws.t, view(ws.Aζ, :, :, j), ws.X′[:, j])
            @. ws.ES′ += ws.t
        end
    end
    @inbounds for d in 1:N
        acc = 0.0
        @simd for j in 1:J
            acc += ws.tr_share[d, j] * ws.X′[d, j]
        end
        lw = ws.has_population ? sc.L̂[d] * ws.ŵ[d] : ws.ŵ[d]
        ws.I′[d] = acc + (ws.has_export_subsidy ? ws.ES′[d] : 0.0) +
                   lw * b.VA[d] - ws.D′[d] + ws.T′[d]
    end
    @inbounds for j in 1:J
        @views mul!(ws.Y′[:, j], view(ws.A, :, :, j), ws.X′[:, j])
    end
    return ws
end
