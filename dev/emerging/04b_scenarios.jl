# 04b_scenarios.jl — stage 2: counterfactuals on the cached baseline.
#
#     julia --project=. dev/emerging/04b_scenarios.jl [which]
#
# `which` selects a subset: us_china, afcfta, chkw, all (default).

using KITE, DataFrames, Printf, Statistics, Serialization

const DIR = joinpath(@__DIR__, "build", "2023")
const WHICH = isempty(ARGS) ? "all" : ARGS[1]
want(s) = WHICH == "all" || WHICH == s

# The package default vfactor = 0.2 is above the stability limit of the wage iteration
# at 196 regions: it fails to converge for a 1 pp tariff just as it does for 25 pp, and
# winsorising the elasticity tail does not help. 0.05 is stable and, being stable, also
# the fastest — 71 outer iterations against 1000 wasted ones. See 04c_stability.jl.
const VFACTOR = 0.05

step(s) = (println("\n", "="^72, "\n", s, "\n", "="^72); flush(stdout))

step("loading cached baseline")
b = deserialize(joinpath(DIR, "baseline.jls"))
@printf("%d countries x %d sectors\n", b.N, b.J); flush(stdout)

iUS = findfirst(==("USA"), b.countries)
iCN = findfirst(==("CHN"), b.countries)

if want("us_china")
    step("United States raises tariffs on China by 25 pp")
    sc = Scenario(b; label = "US +25pp on CHN")
    set_tariff!(sc, b, 0.25; from = "CHN", to = "USA", mode = :add)
    t = @elapsed res = update_equilibrium(CaliendoParro2015(), b, sc; vfactor = VFACTOR)
    @printf("wall %.1f s | iterations %d | criterion %.3e | converged %s\n",
            t, res.iterations, res.criterion, res.converged); flush(stdout)
    df = results(res; level = :country)
    df.welfare_pct = 100 .* (df.welfare_change .- 1)
    sort!(df, :welfare_pct)
    println("\nbiggest losers:");  show(stdout, MIME("text/plain"),
        first(df[!, [:country, :wage_change, :price_index_change, :welfare_pct]], 8))
    println("\n\nbiggest gainers:"); show(stdout, MIME("text/plain"),
        last(df[!, [:country, :wage_change, :price_index_change, :welfare_pct]], 8))
    @printf("\n\nUSA welfare %+.4f%%   CHN welfare %+.4f%%\n",
            df[df.country .== "USA", :welfare_pct][1],
            df[df.country .== "CHN", :welfare_pct][1])
    tradeable = [j for j in 1:b.J if b.π[iCN, iUS, j] > 1e-6]
    @printf("CHN->USA trade share, median pi'/pi over %d sectors with trade: %.3f\n",
            length(tradeable),
            median(res.π′[iCN, iUS, tradeable] ./ b.π[iCN, iUS, tradeable]))
    flush(stdout)
end

if want("afcfta")
    step("AfCFTA — tariffs eliminated among African members")
    africa = [c for c in b.countries if c in Set([
      "DZA","AGO","BEN","BWA","BFA","BDI","CPV","CMR","CAF","TCD","COM","COG","COD","CIV",
      "DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB","GHA","GIN","GNB","KEN","LSO","LBR",
      "LBY","MDG","MWI","MLI","MRT","MUS","MAR","MOZ","NAM","NER","NGA","RWA","STP","SEN",
      "SYC","SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ZMB","ZWE"])]
    @printf("African members in the baseline: %d\n", length(africa)); flush(stdout)
    sc = Scenario(b; label = "AfCFTA")
    set_tariff!(sc, b, 1.0; from = africa, to = africa, mode = :set)
    t = @elapsed res = update_equilibrium(CaliendoParro2015(), b, sc; vfactor = VFACTOR)
    @printf("wall %.1f s | iterations %d | criterion %.3e | converged %s\n",
            t, res.iterations, res.criterion, res.converged); flush(stdout)
    df = results(res; level = :country)
    df.welfare_pct = 100 .* (df.welfare_change .- 1)
    df.african = in.(df.country, Ref(Set(africa)))
    @printf("mean welfare change: African members %+.4f%% | rest of world %+.4f%%\n",
            mean(df[df.african, :welfare_pct]), mean(df[.!df.african, :welfare_pct]))
    println("\nlargest African gains:"); show(stdout, MIME("text/plain"),
        first(sort(df[df.african, [:country, :welfare_pct, :wage_change,
                                   :price_index_change]], :welfare_pct, rev = true), 12))
    flush(stdout)
end

if want("chkw")
    step("CHKW2022 with an empty coalition must reproduce CP2015")
    sc = Scenario(b; label = "empty coalition")
    set_tariff!(sc, b, 0.25; from = "CHN", to = "USA", mode = :add)
    t1 = @elapsed r_cp = update_equilibrium(CaliendoParro2015(), b, sc; vfactor = VFACTOR)
    @printf("CP2015   %.1f s, %d iterations\n", t1, r_cp.iterations); flush(stdout)
    t2 = @elapsed r_ch = update_equilibrium(ChowdhryHinzKaminWanner2022(), b, sc; vfactor = VFACTOR)
    @printf("CHKW2022 %.1f s, %d iterations\n", t2, r_ch.iterations); flush(stdout)
    @printf("max |w_CP - w_CHKW| = %.3e\n", maximum(abs, r_cp.ŵ .- r_ch.ŵ))
end

println("\ndone."); flush(stdout)
