# Cross-validate KITE.jl against the R implementation.
#
# The R solver is used as an independent second implementation of the same equations, but two
# of its defects have to be patched out first or the comparison is meaningless:
#
#   1. The trade-elasticity exponents are inverted — R applies ^(-1/θ) inside the price sum and
#      ^(-θ) outside, the inverse of Caliendo & Parro eq. (11)/(12). The system stays internally
#      consistent (Σ_o π′ = 1 still holds), so it silently solves a model with elasticity 1/θ.
#   2. `cast_variable()` builds arrays positionally from long tables, which recycles values when
#      a table is sparse. We sidestep it by handing R complete Cartesian grids.
#
# Everything else is left untouched, so agreement is genuine evidence for both implementations.
#
# Usage:  Rscript dev/validate_against_R.R [R-KITE-path] [fixture-dir] [output-csv]

suppressMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
rkite <- if (length(args) >= 1) args[1] else "~/Documents/R/KITE"
fixture <- if (length(args) >= 2) args[2] else "test/fixtures/toy_3x2"
outfile <- if (length(args) >= 3) args[3] else "test/fixtures/golden_cp2015_3x2.csv"

for (f in list.files(file.path(path.expand(rkite), "R"), full.names = TRUE)) source(f)

# ── patch 1: restore the published exponent convention ────────────────────────────────────
update_price_index_cp_2015 <- function(price_change, trade_share, trade_cost_change,
                                       input_cost_change, trade_elasticity, model_dimensions) {
    for (s in model_dimensions[["sector"]]) {
        price_change[, s] <-
            ((t(trade_share[, , s]) * (t(trade_cost_change[, , s])^(-trade_elasticity[s]))) %*%
             (input_cost_change[, s]^(-trade_elasticity[s])))^(-1 / trade_elasticity[s])
    }
    price_change[!is.finite(price_change) | price_change == 0] <- 1
    price_change
}

update_trade_share_cp_2015 <- function(trade_share, trade_cost_change, input_cost_change,
                                       price_change, trade_elasticity, model_dimensions) {
    out <- trade_share
    for (s in model_dimensions[["sector"]]) {
        out[, , s] <- t(t(trade_cost_change[, , s] * input_cost_change[, s]) /
                        price_change[, s])^(-trade_elasticity[s]) * trade_share[, , s]
    }
    out
}

# ── read the fixture and rebuild complete grids ───────────────────────────────────────────
rd <- function(name) {
    p <- file.path(fixture, paste0(name, ".csv"))
    if (!file.exists(p)) return(NULL)
    x <- fread(p)
    if (nrow(x) == 0) NULL else x
}

ts <- rd("trade_share")
countries <- sort(unique(c(ts$origin, ts$destination)))
sectors <- sort(unique(ts$sector))

grid3 <- function(x, fill) {
    g <- CJ(origin = countries, destination = countries, sector = sectors, sorted = FALSE)
    if (is.null(x)) { g[, value := fill]; return(g[]) }
    setkey(x, origin, destination, sector)
    g[, value := x[.(g$origin, g$destination, g$sector), value]]
    g[is.na(value), value := fill][]
}

trade_share <- grid3(ts, 0)
tariff <- grid3(rd("tariff"), 1)
export_subsidy <- grid3(rd("export_subsidy"), 1)

ic <- list(
    trade_share        = trade_share,
    intermediate_share = rd("intermediate_share"),
    factor_share       = rd("factor_share"),
    consumption_share  = rd("consumption_share"),
    expenditure        = rd("expenditure"),
    value_added        = rd("value_added"),
    trade_balance      = rd("trade_balance"),
    tariff             = tariff,
    export_subsidy     = export_subsidy,
    elasticities       = list(trade_elasticity = rd("trade_elasticity"))
)

settings <- list(verbose = 0L, tolerance = 1e-12, vfactor = 0.2,
                 max_iterations = 20000, tolerance_output = 1e-14,
                 convergence_method = "root_mean_square")

# ── scenarios ─────────────────────────────────────────────────────────────────────────────
scenarios <- list(
    null = list(),
    uniform_tariff = local({
        t <- copy(tariff); t[origin != destination, value := 1.10]; list(tariff_new = t)
    }),
    bilateral_tariff = local({
        t <- copy(tariff)
        t[origin == countries[2] & destination == countries[1], value := 1.25]
        list(tariff_new = t)
    })
)

out <- rbindlist(lapply(names(scenarios), function(nm) {
    r <- update_equilibrium(caliendo_parro_2015, ic, scenarios[[nm]], settings)
    w <- as.data.table(r$output$wage_change)
    setnames(w, c("country", "wage_change"))
    p <- as.data.table(r$output$price_change)
    setnames(p, c("country", "sector", "price_change"))
    ts_new <- as.data.table(r$output$trade_share_new)
    setnames(ts_new, c("origin", "destination", "sector", "trade_share_new"))

    cat(sprintf("%-18s iterations = %4d, criterion = %.2e\n",
                nm, r$info$iterations, r$info$criterion))

    rbind(
        data.table(scenario = nm, variable = "wage_change", i = w$country, j = "", k = "",
                   value = w$wage_change),
        data.table(scenario = nm, variable = "price_change", i = p$country, j = p$sector,
                   k = "", value = p$price_change),
        data.table(scenario = nm, variable = "trade_share_new", i = ts_new$origin,
                   j = ts_new$destination, k = ts_new$sector, value = ts_new$trade_share_new)
    )
}))

fwrite(out, outfile)
cat("\nwrote", outfile, "-", nrow(out), "rows\n")
