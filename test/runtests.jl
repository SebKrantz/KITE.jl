using KITE
using DataFrames
using CSV
using Test

# ---------------------------------------------------------------------------
# Deterministic synthetic economies. No randomness, no data files: the
# generator is a closed-form hash so a failure is always reproducible.
# ---------------------------------------------------------------------------
function toy_inputs(; N = 4, J = 3, seed = 1)
    countries = ["c$i" for i in 1:N]
    sectors = ["s$j" for j in 1:J]
    h(a, b, c) = 0.3 + 0.7 * ((a * 7 + b * 13 + c * 29 + seed * 5) % 11) / 10
    return (countries = countries, sectors = sectors,
            π = [h(o, d, j) for o in 1:N, d in 1:N, j in 1:J],
            γ = [h(d, k + 3, j + 5) for d in 1:N, k in 1:J, j in 1:J],
            β = [0.25 + 0.3 * ((d * 3 + j * 7 + seed) % 7) / 7 for d in 1:N, j in 1:J],
            θ = [3.0 + ((j * 11 + seed) % 9) for j in 1:J],
            X = [100.0 + 50 * ((d * 5 + j * 3 + seed) % 9) / 9 for d in 1:N, j in 1:J])
end

toy_baseline(; kwargs...) = (t = toy_inputs(; kwargs...);
    calibrate(; countries = t.countries, sectors = t.sectors, π = t.π, γ = t.γ,
                β = t.β, θ = t.θ, X = t.X, verbose = 0))

# symmetric world: every country identical, so a symmetric shock must be felt identically
function symmetric_baseline(; N = 3, J = 2)
    countries = ["c$i" for i in 1:N]
    sectors = ["s$j" for j in 1:J]
    π = [o == d ? 0.6 : 0.4 / (N - 1) for o in 1:N, d in 1:N, j in 1:J]
    γ = fill(1.0 / J, N, J, J)
    β = fill(0.4, N, J)
    return calibrate(; countries = countries, sectors = sectors, π = π, γ = γ, β = β,
                       θ = fill(5.0, J), X = fill(100.0, N, J), verbose = 0)
end

const TIGHT = (verbose = 0, tolerance = 1e-12, inner_tolerance = 1e-14,
               max_iterations = 20_000)
const FIXTURE = joinpath(@__DIR__, "fixtures", "toy_3x2")

@testset "KITE.jl" begin

    @testset "calibration" begin
        b = toy_baseline()
        res = residuals(b)
        @test res.goods_market < 1e-12
        @test res.expenditure < 1e-12
        @test res.income < 1e-12

        # shares are proper shares
        @test all(≈(1.0), sum(b.π, dims = 1))
        @test all(≈(1.0), sum(b.α, dims = 2))
        @test all(x -> x ≥ -eps(), b.α)
        @test all(x -> 0 < x < 1, b.β)

        # the accounting identities hold in levels
        @test sum(b.β .* b.Y, dims = 2) ≈ reshape(b.VA, :, 1)
        @test b.I ≈ b.VA .+ b.R .- b.D
        @test isapprox(sum(b.D), 0.0, atol = 1e-8 * sum(b.VA))

        # calibrating an already-calibrated baseline is a fixed point
        b2 = calibrate(; countries = b.countries, sectors = b.sectors, π = b.π, γ = b.γ,
                         β = b.β, θ = b.θ, X = b.X, τ = b.τ, ζ = b.ζ, verbose = 0)
        @test b2.X ≈ b.X
        @test b2.α ≈ b.α
        @test b2.D ≈ b.D

        # the constructor refuses an inconsistent baseline
        @test_throws ErrorException KiteBaseline(b.countries, b.sectors, b.π, b.γ, b.α, b.β,
                                                 b.θ, b.τ, b.ζ, b.X .* 1.5, b.Y, b.I, b.R,
                                                 b.VA, b.D)
    end

    # -----------------------------------------------------------------------
    # The headline invariant: with no shock, every hat is one and the solver
    # recognises it immediately. This fails the moment the baseline is not
    # model-consistent or the inner loop is not warm-started at it.
    # -----------------------------------------------------------------------
    @testset "no-change scenario is exact" begin
        for b in (toy_baseline(), toy_baseline(N = 6, J = 4, seed = 3), symmetric_baseline())
            for model in (CaliendoParro2015(), ChowdhryHinzKaminWanner2022())
                r = update_equilibrium(model, b; TIGHT...)
                @test r.converged
                @test r.iterations == 1
                @test r.ŵ ≈ ones(b.N) atol = 1e-12
                @test r.P̂ ≈ ones(b.N, b.J) atol = 1e-12
                @test r.π′ ≈ b.π atol = 1e-12
                @test r.X′ ≈ b.X rtol = 1e-12
                @test r.Y′ ≈ b.Y rtol = 1e-12
                @test r.VA′ ≈ b.VA rtol = 1e-12
                @test r.I′ ≈ b.I rtol = 1e-12
                @test welfare_change(r) ≈ ones(b.N) atol = 1e-12
            end
        end
    end

    @testset "trade elasticity" begin
        b = toy_baseline()

        # partial elasticity: dln π / dln τ ≈ −θ(1 − π) for a small shock
        ε = 1e-5
        sc = Scenario(b); set_tariff!(sc, b, 1 + ε; from = "c2", to = "c1", sector = "s1")
        r = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...)
        measured = log(r.π′[2, 1, 1] / b.π[2, 1, 1]) / log1p(ε)
        predicted = -b.θ[1] * (1 - b.π[2, 1, 1])
        @test measured ≈ predicted rtol = 0.1
        @test measured < -1     # the magnitude is θ-like, not 1/θ-like

        # monotonicity in θ — the direct regression test for the inverted exponents in the
        # R implementation, where this comparison reverses
        t = toy_inputs()
        b_hi = calibrate(; countries = t.countries, sectors = t.sectors, π = t.π, γ = t.γ,
                           β = t.β, θ = 2 .* t.θ, X = t.X, verbose = 0)
        shock(bb) = (s = Scenario(bb); set_tariff!(s, bb, 1.2; from = "c2", to = "c1"); s)
        d_lo = let rr = update_equilibrium(CaliendoParro2015(), b, shock(b); TIGHT...)
            abs(rr.π′[2, 1, 1] - b.π[2, 1, 1]) end
        d_hi = let rr = update_equilibrium(CaliendoParro2015(), b_hi, shock(b_hi); TIGHT...)
            abs(rr.π′[2, 1, 1] - b_hi.π[2, 1, 1]) end
        @test d_hi > d_lo
    end

    @testset "tariff comparative statics" begin
        b = toy_baseline()
        sc = Scenario(b); set_tariff!(sc, b, 1.2; from = "c2", to = "c1")
        r = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...)
        @test r.converged

        @test r.π′[2, 1, 1] < b.π[2, 1, 1]        # taxed flow shrinks
        @test r.π′[1, 1, 1] > b.π[1, 1, 1]        # home share rises
        @test r.P̂[1, 1] > 1                       # importer's price index rises
        tr, tr_new = tariff_revenue(r)
        @test tr_new[1] > tr[1]

        # trade diversion: some untaxed origin gains share in that market
        @test any(o -> r.π′[o, 1, 1] > b.π[o, 1, 1], 3:b.N)
    end

    @testset "equilibrium conditions at the solution" begin
        b = toy_baseline()
        sc = Scenario(b); set_tariff!(sc, b, 1.3; from = "c2", to = :all)
        r = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...)

        @test all(≈(1.0), sum(r.π′, dims = 1))
        @test vec(sum(b.β .* r.Y′, dims = 2)) ≈ r.VA′

        # expenditure identity
        ID = zeros(b.N, b.J)
        for j in 1:b.J, k in 1:b.J
            @views ID[:, k] .+= b.input_share[:, k, j] .* r.Y′[:, j]
        end
        @test r.X′ ≈ b.α .* r.I′ .+ ID rtol = 1e-8

        # Walras: aggregate excess demand vanishes
        ta = trade_aggregates(r)
        walras = ta.exports_new .- ta.imports_new .- r.D′ .+ r.T′
        @test isapprox(sum(walras), 0.0, atol = 1e-6 * sum(b.VA))
        @test maximum(abs, walras) < 1e-6 * sum(b.VA)
    end

    @testset "homogeneity of degree one" begin
        t = toy_inputs()
        b1 = toy_baseline()
        b2 = calibrate(; countries = t.countries, sectors = t.sectors, π = t.π, γ = t.γ,
                         β = t.β, θ = t.θ, X = 1000 .* t.X, verbose = 0)
        s1 = Scenario(b1); set_tariff!(s1, b1, 1.2; from = "c2", to = "c1")
        s2 = Scenario(b2); set_tariff!(s2, b2, 1.2; from = "c2", to = "c1")
        r1 = update_equilibrium(CaliendoParro2015(), b1, s1; TIGHT...)
        r2 = update_equilibrium(CaliendoParro2015(), b2, s2; TIGHT...)
        @test r1.ŵ ≈ r2.ŵ atol = 1e-10
        @test r1.π′ ≈ r2.π′ atol = 1e-10
        @test welfare_change(r1) ≈ welfare_change(r2) atol = 1e-10
    end

    # Real quantities are numéraire-invariant only when the trade balance scales with the
    # price level. Under :fixed the deficit is exogenous in nominal terms and legitimately
    # anchors the price level — a property of the model, not of the solver.
    @testset "numéraire invariance" begin
        b = toy_baseline()
        for rule in (:zero, :fixed_global_share)
            sc = Scenario(b); set_tariff!(sc, b, 1.2; from = "c2", to = "c1")
            ra = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...,
                                    trade_balance_rule = rule)
            rb = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...,
                                    trade_balance_rule = rule, numeraire = "c1")
            @test rb.ŵ[1] ≈ 1.0 atol = 1e-10
            scale = ra.ŵ ./ rb.ŵ
            @test maximum(scale) - minimum(scale) < 1e-9        # a common scalar
            @test ra.π′ ≈ rb.π′ atol = 1e-9
            @test welfare_change(ra) ≈ welfare_change(rb) atol = 1e-9
        end
    end

    @testset "trade balance rules" begin
        b = toy_baseline()
        sc = Scenario(b); set_tariff!(sc, b, 1.2; from = "c2", to = "c1")
        @test update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...,
                                 trade_balance_rule = :fixed).D′ ≈ b.D
        @test all(==(0.0), update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...,
                                              trade_balance_rule = :zero).D′)
        r = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...,
                               trade_balance_rule = :fixed_global_share)
        @test r.D′ ./ sum(r.VA′) ≈ b.D ./ sum(b.VA)
        r = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...,
                               trade_balance_rule = :fixed_country_share)
        @test r.D′ ./ r.VA′ ≈ b.D ./ b.VA
    end

    @testset "inner solvers agree" begin
        b = toy_baseline()
        sc = Scenario(b); set_tariff!(sc, b, 1.25; from = "c2", to = "c1")
        ri = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT..., inner_solver = :iterative)
        rd = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT..., inner_solver = :direct)
        @test ri.ŵ ≈ rd.ŵ atol = 1e-9
        @test ri.X′ ≈ rd.X′ rtol = 1e-9
        @test ri.Y′ ≈ rd.Y′ rtol = 1e-9
    end

    # The strongest structural check: the two models share every equation except the wage
    # update, so with no coalition they must land on the same equilibrium.
    @testset "CHKW without a coalition equals CP2015" begin
        b = toy_baseline()
        sc = Scenario(b); set_tariff!(sc, b, 1.2; from = "c2", to = "c1")
        r1 = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...)
        r2 = update_equilibrium(ChowdhryHinzKaminWanner2022(), b, sc; TIGHT..., vfactor = 0.05)
        @test r2.converged
        @test all(==(0.0), r2.T′)
        @test r1.ŵ ≈ r2.ŵ rtol = 1e-6
        @test r1.π′ ≈ r2.π′ rtol = 1e-6
        @test welfare_change(r1) ≈ welfare_change(r2) rtol = 1e-6
    end

    @testset "CHKW transfers" begin
        b = toy_baseline()
        sc = Scenario(b)
        set_ntb!(sc, b, 1.5; from = "c4", to = :all)
        set_coalition!(sc, b, ["c1", "c2"])
        r = update_equilibrium(ChowdhryHinzKaminWanner2022(), b, sc; TIGHT..., vfactor = 0.05)
        @test r.converged

        members = [1, 2]
        others = [3, 4]
        @test isapprox(sum(r.T′[members]), 0.0, atol = 1e-8 * sum(b.VA))
        @test all(==(0.0), r.T′[others])
        @test !all(==(0.0), r.T′[members])          # transfers actually bind

        w = welfare_change(r)
        @test maximum(w[members]) - minimum(w[members]) < 1e-8
    end

    @testset "autarky limit" begin
        b = toy_baseline()
        sc = Scenario(b)
        for o in 1:b.N, d in 1:b.N
            o == d && continue
            set_ntb!(sc, b, 1e5; from = b.countries[o], to = b.countries[d])
        end
        r = update_equilibrium(CaliendoParro2015(), b, sc; verbose = 0, tolerance = 1e-10,
                               max_iterations = 20_000)
        for j in 1:b.J, d in 1:b.N
            @test r.π′[d, d, j] > 0.999
        end
        @test all(welfare_change(r) .< 1.0)         # everyone loses from autarky
    end

    @testset "symmetric world" begin
        b = symmetric_baseline()
        sc = Scenario(b)
        for o in 1:b.N, d in 1:b.N
            o == d || set_tariff!(sc, b, 1.15; from = b.countries[o], to = b.countries[d])
        end
        r = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...)
        @test maximum(r.ŵ) - minimum(r.ŵ) < 1e-9
        w = welfare_change(r)
        @test maximum(w) - minimum(w) < 1e-9
    end

    @testset "convergence criteria" begin
        b = toy_baseline()
        sc = Scenario(b); set_tariff!(sc, b, 1.2; from = "c2", to = "c1")
        ref = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...)
        for m in (:aggregate, :element_wise, :sample)
            r = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT..., convergence = m)
            @test r.ŵ ≈ ref.ŵ rtol = 1e-5
        end
        # :sample is deterministic — the R implementation draws a random subset
        a = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT..., convergence = :sample)
        c = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT..., convergence = :sample)
        @test a.ŵ == c.ŵ
        @test a.iterations == c.iterations
    end

    @testset "results tables" begin
        b = toy_baseline()
        sc = Scenario(b); set_tariff!(sc, b, 1.2; from = "c2", to = "c1")
        r = update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...)

        cdf = results(r; level = :country)
        @test nrow(cdf) == b.N
        @test cdf.welfare_change ≈ cdf.income_change ./ cdf.price_index_change
        @test cdf.real_wage_change ≈ cdf.wage_change ./ cdf.price_index_change
        @test sum(cdf.weight) ≈ 1.0
        @test cdf.value_added_new ≈ r.ŵ .* b.VA

        sdf = results(r; level = :sector)
        @test nrow(sdf) == b.N * b.J

        bdf = results(r; level = :bilateral, drop_zeros = false)
        @test nrow(bdf) == b.N^2 * b.J

        # bilateral fob flows aggregate to the country-level exports and imports
        agg_x = combine(groupby(bdf, :origin), :trade_flow_fob_new => sum => :v)
        sorted = sort(agg_x, :origin)
        @test sorted.v ≈ sort(cdf, :country).exports_new rtol = 1e-9

        # price_index with the baseline weights reproduces price_index_change
        @test price_index(r) ≈ price_index(r; weights = b.α)
    end

    @testset "scenario construction" begin
        b = toy_baseline()
        sc = Scenario(b)
        @test sc.τ′ == b.τ
        set_tariff!(sc, b, 1.5; from = "c1", to = "c2", sector = "s1")
        @test sc.τ′[1, 2, 1] == 1.5
        @test sc.τ′[2, 1, 1] == b.τ[2, 1, 1]        # only the selected cell moved

        set_tariff!(sc, b, 2.0; from = "c1", to = "c2", sector = "s1", mode = :multiply)
        @test sc.τ′[1, 2, 1] == 3.0

        sc2 = Scenario(b)
        set_ntb!(sc2, b, 1.3; from = ["c1", "c2"], to = "c3")
        @test sc2.κ̂[1, 3, 1] == 1.3 && sc2.κ̂[2, 3, 1] == 1.3
        @test sc2.κ̂[3, 3, 1] == 1.0

        set_coalition!(sc2, b, ["c1", "c3"])
        @test sc2.coalition == [true, false, true, false]
        set_coalition!(sc2, b, "c2")
        @test sc2.coalition == [false, true, false, false]
    end

    @testset "option validation" begin
        b = toy_baseline()
        @test_throws ErrorException SolverSettings(convergence = :nonsense)
        @test_throws ErrorException SolverSettings(trade_balance_rule = :nonsense)
        @test_throws ErrorException SolverSettings(inner_solver = :nonsense)
        @test_throws ErrorException SolverSettings(numeraire = :nonsense)
        @test_throws ErrorException SolverSettings(vfactor = 0.0)
        @test_throws ErrorException SolverSettings(vfactor = 1.5)
        @test_throws ErrorException SolverSettings(tolerance = -1)

        r = update_equilibrium(CaliendoParro2015(), b; TIGHT...)
        @test_throws ErrorException results(r; level = :nonsense)
        @test_throws ErrorException calibrate(; countries = b.countries, sectors = b.sectors,
                                                π = b.π, γ = b.γ, β = b.β, θ = b.θ, X = b.X,
                                                anchor = :nonsense, verbose = 0)
        @test_throws ErrorException set_tariff!(Scenario(b), b, 1.1; from = "nowhere")
        @test_throws ErrorException set_tariff!(Scenario(b), b, 1.1; mode = :nonsense)
        @test_throws ErrorException set_tariff!(Scenario(b), b, -1.0)
        @test_throws ErrorException update_equilibrium(CaliendoParro2015(), b;
                                                       TIGHT..., numeraire = "nowhere")
        @test_throws DimensionMismatch KiteBaseline(b.countries, b.sectors, b.π, b.γ, b.α,
                                                    b.β, b.θ, b.τ, b.ζ, b.X[:, 1:1], b.Y,
                                                    b.I, b.R, b.VA, b.D)
    end

    # -----------------------------------------------------------------------
    # Loading: values must be placed by label, never by position. A sparse
    # table is the norm for real MRIO data, and reshaping it positionally is
    # what corrupts 99.9% of the R implementation's trade-share array.
    # -----------------------------------------------------------------------
    @testset "loading and round-trip" begin
        b = read_baseline_csv(FIXTURE; verbose = 0)
        @test b.N == 3 && b.J == 2
        r = update_equilibrium(CaliendoParro2015(), b; TIGHT...)
        @test r.iterations == 1
        @test r.ŵ ≈ ones(b.N) atol = 1e-12

        mktempdir() do dir
            write_baseline(b, dir)
            b2 = read_baseline_csv(dir; verbose = 0)
            @test b2.π ≈ b.π
            @test b2.γ ≈ b.γ
            @test b2.X ≈ b.X
            @test b2.α ≈ b.α
            @test b2.D ≈ b.D

            write_baseline_binary(b, dir)
            b3 = read_baseline_binary(dir)
            @test b3.π == b.π          # binary is bit-exact
            @test b3.X == b.X
            @test b3.θ == b.θ
            @test b3.countries == b.countries
        end

        # a sparse table must be filled by label, not recycled
        mktempdir() do dir
            write_baseline(b, dir)
            full = read_baseline_csv(dir; verbose = 0)
            rows = readlines(joinpath(dir, "trade_share.csv"))
            open(joinpath(dir, "trade_share.csv"), "w") do io
                println(io, rows[1])
                for line in rows[2:end]
                    endswith(split(line, ',')[3], "s2") && continue   # drop a whole sector
                    println(io, line)
                end
            end
            sparse = read_baseline_csv(dir; verbose = 0)
            # sector s2 becomes a closed market; sector s1 is untouched
            @test sparse.π[:, :, 1] ≈ full.π[:, :, 1]
            @test all(d -> sparse.π[d, d, 2] ≈ 1.0, 1:b.N)
        end

        # duplicate and unknown labels are errors, not silent corruption
        mktempdir() do dir
            write_baseline(b, dir)
            rows = readlines(joinpath(dir, "trade_share.csv"))
            open(joinpath(dir, "trade_share.csv"), "w") do io
                for line in rows; println(io, line); end
                println(io, rows[2])                      # duplicate key
            end
            @test_throws ErrorException read_baseline_csv(dir; verbose = 0)
        end
    end

    @testset "type stability and allocation" begin
        b = toy_baseline()
        sc = Scenario(b); set_tariff!(sc, b, 1.2; from = "c2", to = "c1")
        @test (@inferred update_equilibrium(CaliendoParro2015(), b, sc;
                                            verbose = 0)) isa KiteResult

        # a steady-state outer iteration must not allocate
        ws = KITE._Workspace(b, sc)
        settings = SolverSettings(verbose = 0)
        KITE._input_cost!(ws, b, sc); KITE._price_index!(ws, b)
        KITE._trade_shares!(ws, b);   KITE._policy_weights!(ws, b, sc)
        allocated = @allocated begin
            KITE._input_cost!(ws, b, sc)
            KITE._price_index!(ws, b)
            KITE._trade_shares!(ws, b)
            KITE._policy_weights!(ws, b, sc)
            KITE._value_added!(ws, b)
            KITE._update_trade_balance!(ws, b, :fixed)
            KITE._wage_update!(CaliendoParro2015(), ws, b, sc, settings)
        end
        @test allocated == 0
    end

    # -----------------------------------------------------------------------
    # Cross-validation against the R implementation, used as an independent
    # second implementation of the same equations. Its inverted trade-elasticity
    # exponents are patched out first (see dev/validate_against_R.R), and it is
    # fed complete Cartesian grids so its positional array casting cannot
    # corrupt them. The golden file is regenerated by that script.
    # -----------------------------------------------------------------------
    @testset "cross-validation against R" begin
        golden_path = joinpath(@__DIR__, "fixtures", "golden_cp2015_3x2.csv")
        if !isfile(golden_path)
            @info "golden reference not found; skipping cross-validation."
        else
            b = read_baseline_csv(FIXTURE; verbose = 0)
            g = CSV.read(golden_path, DataFrame)
            g.j = coalesce.(g.j, "")
            g.k = coalesce.(g.k, "")

            function run_scenario(name)
                sc = Scenario(b)
                if name == "uniform_tariff"
                    for o in 1:b.N, d in 1:b.N
                        o == d || set_tariff!(sc, b, 1.10; from = b.countries[o],
                                              to = b.countries[d])
                    end
                elseif name == "bilateral_tariff"
                    set_tariff!(sc, b, 1.25; from = b.countries[2], to = b.countries[1])
                end
                update_equilibrium(CaliendoParro2015(), b, sc; TIGHT...)
            end

            for name in unique(g.scenario)
                r = run_scenario(name)
                for row in eachrow(g[g.scenario .== name, :])
                    julia_value = if row.variable == "wage_change"
                        r.ŵ[b.country_index[row.i]]
                    elseif row.variable == "price_change"
                        r.P̂[b.country_index[row.i], b.sector_index[row.j]]
                    else
                        r.π′[b.country_index[row.i], b.country_index[row.j],
                             b.sector_index[row.k]]
                    end
                    @test julia_value ≈ row.value rtol = 1e-9
                end
            end
        end
    end

    @testset "convergence criterion helper" begin
        old = [1.0, 2.0, 3.0, 4.0]
        new = [1.01, 2.02, 2.99, 4.01]
        for m in (:root_mean_square, :aggregate, :element_wise, :sample)
            @test KITE._criterion(new, old, m) > 0
            @test KITE._criterion(old, old, m) == 0
        end
        @test KITE._criterion(zeros(3), zeros(3), :aggregate) == 0
        @test KITE._criterion(Float64[], Float64[], :root_mean_square) == 0
        @test_throws ErrorException KITE._criterion(new, old, :nonsense)
        @test_throws DimensionMismatch KITE._criterion(new, old[1:2], :element_wise)
        @test KITE._criterion([1.1, 2.0], [1.0, 2.0], :element_wise) ≈ 0.1 rtol = 1e-12
    end
end
