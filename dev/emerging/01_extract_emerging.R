# 01_extract_emerging.R — EMERGING V2 MRIO -> KITE initial-condition tables.
#
# The transaction matrix z is 32585 x 32585 (8.5 GB dense). We never hold it: the
# HDF5 chunk layout stores one column segment per chunk, so reading the 133
# consecutive columns that make up everything one destination economy buys is
# chunk-aligned and cheap. Each block is reduced immediately and the 245 economies
# are pooled to the KITE regions on the fly, so peak memory stays under 1 GB.
#
# Row/column index of the MRIO is country-major: position (c-1)*133 + s.
#
# Writes long CSVs to dev/emerging/build/<year>/.

source("dev/emerging/00_common.R")
suppressPackageStartupMessages(library(rhdf5))

MAT <- file.path(EMERGING_DIR, "V2", sprintf("EMERGING_V2_%d_m.mat", YEAR))
stopifnot(file.exists(MAT))
DEST <- file.path(OUT_DIR, as.character(YEAR))
dir.create(DEST, showWarnings = FALSE, recursive = TRUE)
NBLOCK <- if (nzchar(Sys.getenv("KITE_NBLOCK"))) as.integer(Sys.getenv("KITE_NBLOCK")) else NC_RAW

msg("EMERGING V2 %d -> KITE tables (%d economies x %d sectors -> %d x %d)",
    YEAR, NC_RAW, NS, NC, NS)

sec_g <- qF(rep.int(seq_len(NS), NC_RAW))     # supplying sector of each MRIO row
t0 <- Sys.time()

# ── value added and gross output ──────────────────────────────────────────────
va  <- drop(h5read(MAT, "va"))
out <- drop(h5read(MAT, "X"))
stopifnot(length(va) == NC_RAW * NS, length(out) == NC_RAW * NS)

agg_c <- function(m) t(fsum(t(m), CMAP))      # [sector, 245] -> [sector, 196]
VA_sec  <- agg_c(matrix(va,  NS, NC_RAW))
OUT_sec <- agg_c(matrix(out, NS, NC_RAW))

# ── final demand: 32585 x 735 = (origin,sector) x (destination x 3 categories) ─
msg("reading final demand ...")
fd <- h5read(MAT, "f")
stopifnot(dim(fd)[1] == NC_RAW * NS, dim(fd)[2] == NC_RAW * 3L)

# ── stream z one destination economy at a time ────────────────────────────────
ABS    <- array(0, c(NS, NC, NC))   # [j, o, d] absorption by d of sector-j goods from o
ZS     <- array(0, c(NS, NS, NC))   # [j, k, d] d's use of sector-j goods as input to sector k
FD_sec <- matrix(0, NS, NC)         # [j, d]    d's final demand for sector-j goods
zrow   <- numeric(NC_RAW * NS)      # row totals of z, for the gross-output identity check

fid <- H5Fopen(MAT, flags = "H5F_ACC_RDONLY")
for (d in seq_len(NBLOCK)) {
  cols <- ((d - 1L) * NS + 1L):(d * NS)
  M    <- h5read(fid, "z", index = list(NULL, cols))          # 32585 x 133
  Fd   <- fd[, ((d - 1L) * 3L + 1L):(d * 3L), drop = FALSE]   # 32585 x 3

  mrow <- .rowSums(M,  nrow(M),  NS)
  frow <- .rowSums(Fd, nrow(Fd), 3L)
  zrow <- zrow + mrow

  k <- CMAP[d]
  ABS[, , k]  <- ABS[, , k] + agg_c(matrix(mrow + frow, NS, NC_RAW))
  ZS[, , k]   <- ZS[, , k]  + fsum(M, sec_g)                  # [supplying j, using k]
  FD_sec[, k] <- FD_sec[, k] + .rowSums(matrix(frow, NS, NC_RAW), NS, NC_RAW)

  if (d %% 25L == 0L) msg("  ... %d/%d economies (%.1f min)", d, NBLOCK,
                          as.numeric(difftime(Sys.time(), t0, units = "mins")))
}
H5Fclose(fid)
frow_all <- .rowSums(fd, nrow(fd), ncol(fd))
rm(fd); gc()
msg("stream complete in %.1f min", as.numeric(difftime(Sys.time(), t0, units = "mins")))

# ── derive KITE variables ─────────────────────────────────────────────────────
X_mat <- t(apply(ABS, c(1, 3), sum))                     # expenditure [d, j]

PI <- aperm(ABS, c(2, 3, 1))                             # [o, d, j]
for (j in seq_len(NS)) {
  den <- X_mat[, j]; pos <- den > 0
  if (any(pos))  PI[, pos, j] <- sweep(PI[, pos, j, drop = FALSE], 2L, den[pos], "/")
  if (any(!pos)) { PI[, !pos, j] <- 0
                   for (d in which(!pos)) PI[d, d, j] <- 1 }   # close empty markets to home
}

GAM <- ZS                                                # [input j, output k, country d]
n_gam_empty <- 0L
for (d in seq_len(NC)) {
  cs <- .colSums(ZS[, , d], NS, NS); ok <- cs > 0
  if (any(ok))  GAM[, ok, d] <- sweep(ZS[, ok, d, drop = FALSE], 2L, cs[ok], "/")
  if (any(!ok)) {
    # sector buys no intermediates at all: point the (near-zero weight) bundle at
    # itself so the input-cost aggregator stays a proper weighted mean
    GAM[, !ok, d] <- 0
    for (k in which(!ok)) GAM[k, k, d] <- 1
    n_gam_empty <- n_gam_empty + sum(!ok)
  }
}

# Gross output on the supply side — what the sector actually sells — rather than the
# reported X vector. The two agree to 5e-8 worldwide but disagree cell by cell, and
# only the supply-side figure is consistent with the pi and X we just built.
GROSS_sec <- agg_c(matrix(zrow + frow_all, NS, NC_RAW))
BETA_raw  <- t(VA_sec / GROSS_sec)                       # [d, j]
n_beta_bad <- sum(!is.finite(BETA_raw) | BETA_raw <= 0 | BETA_raw >= 1)
BETA <- BETA_raw
BETA[!is.finite(BETA)] <- 1 - 1e-4
BETA <- pmin(pmax(BETA, 1e-4), 1 - 1e-4)

ALPHA <- t(sweep(FD_sec, 2L, pmax(.colSums(FD_sec, NS, NC), 1e-12), "/"))
VA_c  <- .colSums(VA_sec, NS, NC)

# ── diagnostics ───────────────────────────────────────────────────────────────
if (NBLOCK == NC_RAW) {
  gross <- zrow + frow_all
  big   <- out > 1e-6
  rel   <- abs(gross[big] / out[big] - 1)
  msg("\ngross output, supply side rowSums(z)+rowSums(f) vs reported X:")
  msg("  world totals: %.0f vs %.0f (rel %.2e)", sum(gross), sum(out),
      abs(sum(gross) / sum(out) - 1))
  msg("  cells with X > 0: %d | rel dev > 1%%: %d | > 50%%: %d | median %.2e",
      sum(big), sum(rel > 0.01), sum(rel > 0.5), median(rel))
}
msg("\nworld value added : %.0f mn USD", sum(VA_c))
msg("world expenditure : %.0f mn USD", sum(X_mat))
msg("max |sum_o pi - 1| : %.3e", max(abs(apply(PI, c(2, 3), sum) - 1)))
msg("max |sum_k gam - 1|: %.3e", max(abs(apply(GAM, c(2, 3), sum) - 1)))
msg("gamma columns with no intermediate use, closed to own sector: %d of %d",
    n_gam_empty, NC * NS)
msg("beta cells outside (0,1) or undefined, clipped: %d of %d", n_beta_bad, NC * NS)
msg("beta range        : [%.4f, %.4f]", min(BETA), max(BETA))
msg("zero-expenditure (region,sector) cells: %d of %d", sum(X_mat <= 0), length(X_mat))

# ── write long CSVs ───────────────────────────────────────────────────────────
w <- function(dt, name) {
  fwrite(dt, file.path(DEST, paste0(name, ".csv")))
  msg("  %-22s %9d rows", paste0(name, ".csv"), nrow(dt))
}
msg("\nwriting long CSVs ...")

pi_dt <- CJ(sector = SECTORS, destination = COUNTRIES, origin = COUNTRIES, sorted = FALSE)
set(pi_dt, j = "value", value = as.vector(PI))
w(pi_dt[value > 0, .(origin, destination, sector, value)], "trade_share")

gam_dt <- CJ(country = COUNTRIES, output = SECTORS, input = SECTORS, sorted = FALSE)
set(gam_dt, j = "value", value = as.vector(GAM))
w(gam_dt[value > 0, .(country, input, output, value)], "intermediate_share")

cs_dt <- data.table(country = rep(COUNTRIES, each = NS), sector = SECTORS)
w(copy(cs_dt)[, value := as.vector(t(BETA))],  "factor_share")
w(copy(cs_dt)[, value := as.vector(t(X_mat))], "expenditure")
w(copy(cs_dt)[, value := as.vector(t(ALPHA))], "consumption_share")
w(data.table(country = COUNTRIES, value = VA_c), "value_added")

saveRDS(list(PI = PI, GAM = GAM, BETA = BETA, BETA_raw = BETA_raw, ALPHA = ALPHA,
             X = X_mat, VA_sec = VA_sec, OUT_sec = OUT_sec, GROSS_sec = GROSS_sec,
             FD_sec = FD_sec, countries = COUNTRIES, sectors = SECTORS, year = YEAR),
        file.path(DEST, "arrays.rds"))

msg("\ndone in %.1f min -> %s", as.numeric(difftime(Sys.time(), t0, units = "mins")), DEST)
