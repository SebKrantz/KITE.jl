# 00_common.R — shared paths, classifications and the country/sector maps.
#
# Sourced by every other script in dev/emerging. Nothing here reads the MRIO itself.

suppressPackageStartupMessages({
  library(data.table)
  library(collapse)
  library(readxl)
})

EMERGING_DIR <- path.expand("~/Documents/Data/EMERGING")
KITE_R_DIR   <- path.expand("~/Documents/R/KITE")
YEAR         <- 2023L

if (!dir.exists("dev/emerging")) stop("run these scripts from the KITE.jl repo root")
OUT_DIR <- "dev/emerging/build"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

NC_RAW <- 245L   # economies in EMERGING V2
NS     <- 133L   # sectors in EMERGING V2

# ── classifications ───────────────────────────────────────────────────────────
EM_CTRY <- qDT(read_xlsx(file.path(EMERGING_DIR, "classification/EMERGING_Country.xlsx")))
EM_SEC  <- qDT(read_xlsx(file.path(EMERGING_DIR, "classification/EMERGING_Sector_V2.xlsx")))
stopifnot(nrow(EM_CTRY) == NC_RAW, nrow(EM_SEC) == NS)

UN <- qDT(africamonitor::am_countries_wld)   # 195 UN-recognised economies

# ── sector codes ──────────────────────────────────────────────────────────────
# EMERGING's 133 sectors are 96 HS-2002 chapters (1-97 less 27 and the unused 77),
# 7 energy/utility sectors carved out of HS chapter 27, and 30 EBOPS services.
SERVICE_CODES <- c(
  "MSPI", "MAIN", "TSEA", "TAIR", "TOTH", "POST", "TRVG", "TLOC", "ACCO", "FOOD",
  "CONS", "INSD", "PENS", "FINS", "REAL", "IPCH", "TELE", "COMP", "INFO", "RSDV",
  "PROF", "ENGI", "WAST", "LEAS", "OBUS", "AUDV", "HLTH", "EDUC", "RECR", "GOVT")
ENERGY_CODES  <- c("ELY", "GASD", "WATR", "COAL", "OIL", "GAS", "PETR")

EM_SEC[, hs2 := as.integer(Code_HS2002)]
EM_SEC[, sector := fifelse(hs2 <= 98L, sprintf("HS%02d", hs2), NA_character_)]
EM_SEC[hs2 %in% 99:105,  sector := ENERGY_CODES]
EM_SEC[hs2 %in% 106:135, sector := SERVICE_CODES]
# HS98 in the EMERGING file is "commodities not specified according to kind"
EM_SEC[hs2 == 98L, sector := "HS98"]
stopifnot(!anyNA(EM_SEC$sector), !anyDuplicated(EM_SEC$sector))

# group flags used by the elasticity and tariff scripts
EM_SEC[, is_hs_chapter := hs2 <= 98L]                 # 96 sectors with a real HS chapter
EM_SEC[, is_service    := hs2 >= 106L]                # 30 EBOPS services
EM_SEC[, is_energy     := hs2 %in% 99:105]            # 7 carved out of HS27

SECTORS <- EM_SEC$sector                              # in EMERGING row order

# ── country map: 195 UN economies kept, everything else pooled into ROW ───────
EM_CTRY[, kite := fifelse(ISO3 %in% UN$ISO3, ISO3, "ROW")]
COUNTRIES <- c(sort(UN$ISO3), "ROW")                  # 196, ROW last
NC <- length(COUNTRIES)
CMAP <- match(EM_CTRY$kite, COUNTRIES)                # 245 -> 196
stopifnot(!anyNA(CMAP), length(unique(CMAP)) == NC)

msg <- function(...) cat(sprintf(...), "\n", sep = "")
