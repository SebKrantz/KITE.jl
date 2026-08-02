# 03_tariffs.R — bilateral applied tariffs for the 133 EMERGING sectors.
#
# MAcMap-HS6 2019 (CEPII, distributed with Fontagne, Guimbard & Orefice 2022) is a
# 3.6 GB CSV of 144 m (importer, exporter, HS6) applied ad-valorem equivalents. It is
# streamed through awk and reduced to chapter means in one pass, never loaded whole.
#
# Because EMERGING's goods sectors are HS chapters, aggregation is exact at the
# concordance level: every HS6 line belongs to exactly one sector. Chapter 27 is the
# one exception and gets a second pass at 4-digit, to feed the seven energy sectors.
# Within a cell the mean over HS6 lines is unweighted — the 2019 release carries no
# trade values — but the cells are 96 chapters rather than 26 industries, so far less
# heterogeneity is averaged over than in the ICIO build.
#
# 152 of 245 economies file an import schedule. Missing importers and exporters are
# filled in two documented tiers, and every cell carries the tier it came from:
#   1. customs union — exact, since members apply a common external tariff
#      (EU, SACU, Switzerland-Liechtenstein, France-Monaco, Italy-San Marino);
#   2. regional median over reporting economies in the same UN region.
# Anything still absent, and every service sector, is 1 (no tariff).
#
# Writes tariff.csv to dev/emerging/build/<year>/.

source("dev/emerging/00_common.R")

TARIFF_ZIP <- file.path(KITE_R_DIR, "data/raw/macmap_tariffs/Tariffs_2001_2019.zip")
CSV_IN_ZIP <- "Tariffs_2001_2019/mmhs6_2019.csv"
CACHE      <- OUT_DIR
DEST       <- file.path(OUT_DIR, as.character(YEAR))
dir.create(DEST, showWarnings = FALSE, recursive = TRUE)

# ── one streaming pass per aggregation level, cached ──────────────────────────
stream_awk <- function(outfile, prog, label) {
  if (file.exists(outfile)) { msg("using cached %s", basename(outfile)); return(invisible()) }
  msg("streaming MAcMap 2019 -> %s (%s) ...", basename(outfile), label)
  cmd <- sprintf("7z e -so %s '%s' 2>/dev/null | awk -F, -v OFS=, '%s' > %s",
                 shQuote(TARIFF_ZIP), CSV_IN_ZIP, prog, shQuote(outfile))
  if (system(cmd) != 0L || file.size(outfile) < 1000) stop("awk pass failed for ", outfile)
}
stream_awk(file.path(CACHE, "macmap_hs2_2019.csv"),
  'NR>1{k=$1 FS $2 FS substr($3,1,2); s[k]+=$5; c[k]++} END{print "importer","exporter","hs","sum_adv","n"; for(k in s) print k, s[k], c[k]}',
  "all chapters, 2-digit")
stream_awk(file.path(CACHE, "macmap_hs4_ch27_2019.csv"),
  'NR>1 && substr($3,1,2)=="27"{k=$1 FS $2 FS substr($3,1,4); s[k]+=$5; c[k]++} END{print "importer","exporter","hs","sum_adv","n"; for(k in s) print k, s[k], c[k]}',
  "chapter 27, 4-digit")

t2 <- fread(file.path(CACHE, "macmap_hs2_2019.csv"))
t4 <- fread(file.path(CACHE, "macmap_hs4_ch27_2019.csv"))
msg("loaded %d chapter cells and %d chapter-27 heading cells", nrow(t2), nrow(t4))

# ── map HS codes to EMERGING sectors ──────────────────────────────────────────
t2 <- t2[hs != 27L]                                   # chapter 27 comes from t4
t2[, sector := sprintf("HS%02d", hs)]
t2 <- t2[sector %in% SECTORS]

hs27_map <- rbindlist(list(
  data.table(hs = c(2701L, 2702L, 2703L, 2704L),                     sector = "COAL"),
  data.table(hs = 2709L,                                             sector = "OIL"),
  data.table(hs = 2711L,                                             sector = "GAS"),
  data.table(hs = c(2706L, 2707L, 2708L, 2710L, 2712L, 2713L, 2714L, 2715L),
                                                                     sector = "PETR"),
  data.table(hs = 2705L,                                             sector = "GASD"),
  data.table(hs = 2716L,                                             sector = "ELY")))
t4 <- join(t4, hs27_map, on = "hs", how = "inner", verbose = 2)

tar <- rbindlist(list(t2[, .(importer, exporter, sector, sum_adv, n)],
                      t4[, .(importer, exporter, sector, sum_adv, n)]))

# ── map economies to KITE regions ─────────────────────────────────────────────
kite_of <- setNames(EM_CTRY$kite, EM_CTRY$ISO3)
tar[, `:=`(d = kite_of[importer], o = kite_of[exporter])]
unknown <- sort(unique(c(tar[is.na(d)]$importer, tar[is.na(o)]$exporter)))
if (length(unknown)) msg("dropping %d economies absent from EMERGING: %s",
                         length(unknown), paste(unknown, collapse = " "))
tar <- tar[!is.na(d) & !is.na(o)]

reg <- tar[, .(adv = sum(sum_adv) / sum(n)), by = .(o, d, sector)]
have_imp <- sort(unique(reg$d)); have_exp <- sort(unique(reg$o))
msg("regions with an import schedule: %d of %d | as exporter: %d",
    length(have_imp), NC, length(have_exp))

# ── tier 1: customs unions ────────────────────────────────────────────────────
EU <- c("AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC","HUN",
        "IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK","SVN","ESP","SWE")
EU_have <- intersect(EU, have_imp)
# donors: a member without its own schedule inherits the union's common external tariff
CU <- rbindlist(list(
  data.table(member = setdiff(c(EU, "AND", "SMR", "MCO"), EU_have), donor = "EU"),
  data.table(member = c("BWA", "LSO", "NAM", "SWZ"),                donor = "ZAF"),
  data.table(member = "LIE",                                        donor = "CHE")))
CU <- CU[member %in% COUNTRIES]

eu_imp <- reg[d %in% EU_have, .(adv = mean(adv)), by = .(o, sector)]
eu_exp <- reg[o %in% intersect(EU, have_exp), .(adv = mean(adv)), by = .(d, sector)]

add_imp <- rbindlist(c(
  lapply(CU[donor == "EU"]$member,  function(m) copy(eu_imp)[, `:=`(d = m)]),
  lapply(CU[donor != "EU"]$member,  function(m) {
    dn <- CU[member == m]$donor
    if (!dn %in% have_imp) return(NULL)
    copy(reg[d == dn, .(o, sector, adv)])[, `:=`(d = m)] })), use.names = TRUE)
add_imp <- add_imp[!is.na(adv) & d %!in% have_imp]

reg <- rbindlist(list(reg, add_imp[, .(o, d, sector, adv)]), use.names = TRUE)
msg("tier 1 (customs union) added %d importer cells for %d regions",
    nrow(add_imp), uniqueN(add_imp$d))

# same treatment on the exporter side
have_exp <- sort(unique(reg$o))
add_exp <- rbindlist(c(
  lapply(CU[donor == "EU"]$member, function(m) copy(eu_exp)[, `:=`(o = m)]),
  lapply(CU[donor != "EU"]$member, function(m) {
    dn <- CU[member == m]$donor
    if (!dn %in% have_exp) return(NULL)
    copy(reg[o == dn, .(d, sector, adv)])[, `:=`(o = m)] })), use.names = TRUE)
add_exp <- add_exp[!is.na(adv) & o %!in% have_exp]
reg <- rbindlist(list(reg, add_exp[, .(o, d, sector, adv)]), use.names = TRUE)
msg("tier 1 (customs union) added %d exporter cells for %d regions",
    nrow(add_exp), uniqueN(add_exp$o))

reg <- reg[, .(adv = mean(adv)), by = .(o, d, sector)]   # collapse any overlap
reg[, tier := "reported or customs union"]

# ── tier 2: regional median over reporting economies ──────────────────────────
region_of <- setNames(UN$Region, UN$ISO3); region_of["ROW"] <- "ROW"
have_imp <- sort(unique(reg$d)); have_exp <- sort(unique(reg$o))
miss_imp <- setdiff(COUNTRIES, have_imp); miss_exp <- setdiff(COUNTRIES, have_exp)

reg[, d_reg := region_of[d]]
reg_med <- reg[d %in% have_imp, .(adv = median(adv)), by = .(d_reg, o, sector)]
world_med <- reg[, .(adv = median(adv)), by = .(o, sector)]

fill_imp <- rbindlist(lapply(miss_imp, function(m) {
  r <- region_of[[m]]
  x <- reg_med[d_reg == r, .(o, sector, adv)]
  if (!nrow(x)) x <- world_med[, .(o, sector, adv)]
  x[, `:=`(d = m, tier = "regional median")][]
}), use.names = TRUE)
reg[, d_reg := NULL]
reg <- rbindlist(list(reg, fill_imp[, .(o, d, sector, adv, tier)]), use.names = TRUE)
msg("tier 2 (regional median) added %d cells for %d importers: %s",
    nrow(fill_imp), length(miss_imp), paste(miss_imp, collapse = " "))

# missing exporters: what the importer typically charges in that sector
exp_med <- reg[, .(adv = median(adv)), by = .(d, sector)]
fill_exp <- rbindlist(lapply(miss_exp, function(m)
  copy(exp_med)[, `:=`(o = m, tier = "regional median")]), use.names = TRUE)
reg <- rbindlist(list(reg, fill_exp[, .(o, d, sector, adv, tier)]), use.names = TRUE)
msg("tier 2 added %d cells for %d exporters: %s",
    nrow(fill_exp), length(miss_exp), paste(miss_exp, collapse = " "))

# ── full grid ─────────────────────────────────────────────────────────────────
grid <- CJ(origin = COUNTRIES, destination = COUNTRIES, sector = SECTORS, sorted = FALSE)
setnames(reg, c("o", "d"), c("origin", "destination"))
full <- join(grid, reg[, .(origin, destination, sector, adv, tier)],
             on = c("origin", "destination", "sector"), how = "left", verbose = 2)

full[is.na(adv), `:=`(adv = 0, tier = "no tariff")]
full[origin == destination, `:=`(adv = 0, tier = "self trade")]
full[, value := 1 + adv]

msg("\ntariff cells by tier:")
print(full[, .(cells = .N, mean_adv = round(mean(adv), 4), max = round(max(adv), 3)), by = tier])
msg("\nmean applied tariff on cross-border goods cells: %.4f",
    full[origin != destination & sector %in% EM_SEC[is_hs_chapter | is_energy]$sector, mean(adv)])
msg("value range: %.3f - %.3f", min(full$value), max(full$value))

fwrite(full[value != 1, .(origin, destination, sector, value)],
       file.path(DEST, "tariff.csv"))
fwrite(full[, .(origin, destination, sector, value, tier)],
       file.path(CACHE, sprintf("tariff_%d_detail.csv", YEAR)))
msg("\nwrote tariff.csv (%d non-unit cells of %d) -> %s",
    nrow(full[value != 1]), nrow(full), DEST)
