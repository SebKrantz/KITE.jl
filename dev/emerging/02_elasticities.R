# 02_elasticities.R — trade elasticities for the 133 EMERGING sectors.
#
# EMERGING's goods sectors ARE HS-2002 chapters, so unlike the ICIO build there is
# no lossy HS -> SITC -> industry concordance: the HS6 estimates of Fontagne,
# Guimbard & Orefice (2022) aggregate straight to the chapter that defines the
# sector. theta = |epsilon|, the tariff elasticity of trade, which is the Frechet
# dispersion parameter in Caliendo & Parro's eq. (11).
#
# Three groups need a rule beyond that:
#   * the 7 energy/utility sectors carved out of HS chapter 27, mapped by 4-digit
#     heading (coal 2701-04, crude oil 2709, gas 2711, refining 2706-08+2710+2712-15,
#     manufactured gas 2705, electricity 2716; water has no HS code);
#   * HS98 ("commodities not specified") and the two chapters whose HS6 lines are
#     all insignificant (45 cork, 67 feathers), which take the goods median;
#   * the 30 EBOPS services, which no tariff-based method can identify. They take a
#     flat theta = SERVICES_THETA. Ahmad & Schreiber (USITC 2024) put the median
#     Armington elasticity of substitution for services at 5.98 across NAICS-3
#     industries (5.42 at GTAP-sector level, 5.01 for core tradable services), and
#     the trade elasticity is sigma - 1.
#
# Writes trade_elasticity.csv to dev/emerging/build/<year>/.

source("dev/emerging/00_common.R")
suppressPackageStartupMessages(library(haven))

ELAST_DTA      <- file.path(KITE_R_DIR, "data/raw/fontagne_elasticities/elast_hs6.dta")
SERVICES_THETA <- 5.0
DEST <- file.path(OUT_DIR, as.character(YEAR))
dir.create(DEST, showWarnings = FALSE, recursive = TRUE)

e <- qDT(read_dta(ELAST_DTA))
msg("elast_hs6.dta: %d HS6 lines", nrow(e))

# keep correctly-signed, non-missing estimates
e <- e[!is.na(epsilon) & positive != 1 & missing != 1]
e[, `:=`(theta = abs(epsilon),
         hs2   = as.integer(substr(HS6, 1, 2)),
         hs4   = as.integer(substr(HS6, 1, 4)))]
msg("usable estimates: %d (median theta %.2f, IQR %.2f-%.2f)",
    nrow(e), median(e$theta), quantile(e$theta, .25), quantile(e$theta, .75))

GOODS_MEDIAN <- median(e[hs2 != 27L]$theta)

# ── HS chapters ───────────────────────────────────────────────────────────────
ch <- e[hs2 != 27L, .(theta = median(theta), n_hs6 = .N), by = hs2]

# ── the 7 sectors carved out of chapter 27 ────────────────────────────────────
hs27_map <- rbindlist(list(
  data.table(hs4 = c(2701L, 2702L, 2703L, 2704L),               sector = "COAL"),
  data.table(hs4 = 2709L,                                       sector = "OIL"),
  data.table(hs4 = 2711L,                                       sector = "GAS"),
  data.table(hs4 = c(2706L, 2707L, 2708L, 2710L,
                     2712L, 2713L, 2714L, 2715L),               sector = "PETR"),
  data.table(hs4 = 2705L,                                       sector = "GASD"),
  data.table(hs4 = 2716L,                                       sector = "ELY")))

en <- join(e[hs2 == 27L], hs27_map, on = "hs4", how = "inner", verbose = 2)[
  , .(theta = median(theta), n_hs6 = .N), by = sector]

# ── assemble the 133-sector table ─────────────────────────────────────────────
te <- EM_SEC[, .(sector, hs2, is_hs_chapter, is_service, is_energy)]
te <- join(te, ch, on = "hs2", how = "left", verbose = 0)
setnames(en, c("theta", "n_hs6"), c("theta_en", "n_en"))
te <- join(te, en, on = "sector", how = "left", verbose = 0)
te[!is.na(theta_en), `:=`(theta = theta_en, n_hs6 = n_en)]
te[, c("theta_en", "n_en") := NULL]

te[is_service == TRUE, `:=`(theta = SERVICES_THETA, n_hs6 = 0L)]
te[sector == "WATR",   `:=`(theta = SERVICES_THETA, n_hs6 = 0L)]   # utility, no HS code
fallback <- te[is.na(theta), sector]
te[is.na(theta), `:=`(theta = GOODS_MEDIAN, n_hs6 = 0L)]

te[, source := fifelse(is_service | sector == "WATR", "services literature",
              fifelse(sector %in% fallback, "goods median",
              fifelse(is_energy, "HS 2007 chapter 27, 4-digit", "HS 2007 chapter")))]

stopifnot(nrow(te) == NS, !anyNA(te$theta), all(te$theta > 0))

msg("\ngoods median theta: %.2f   | sectors on the goods-median fallback: %s",
    GOODS_MEDIAN, paste(fallback, collapse = ", "))
msg("theta range: %.2f - %.2f", min(te$theta), max(te$theta))
print(te[, .(n = .N, min = round(min(theta), 2), median = round(median(theta), 2),
             max = round(max(theta), 2)), by = source])
msg("\nenergy sectors:")
print(te[is_energy == TRUE, .(sector, theta = round(theta, 2), n_hs6, source)])

fwrite(te[, .(sector, value = theta)], file.path(DEST, "trade_elasticity.csv"))
fwrite(te, file.path(DEST, "trade_elasticity_detail.csv"))
msg("\nwrote trade_elasticity.csv (%d sectors) -> %s", nrow(te), DEST)
