# Export the R KITE initial conditions to the long CSV interchange format read by
# `KITE.read_baseline_csv`.
#
# This deliberately does NOT use the R package's own `cast_variable()`. That function builds
# arrays positionally from a sparse long table, which silently recycles values into the wrong
# cells: on the 2022 database 26,179 of 328,050 trade-share cells are absent, and 99.9% of the
# resulting array is wrong. Here every table is written as-is in long form, and KITE.jl scatters
# it by label with an explicit fill.
#
# Usage:
#   Rscript dev/export_baseline_from_R.R <path/to/initial_conditions.rds> <output-dir>

suppressMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
rds <- if (length(args) >= 1) args[1] else
    "~/Documents/R/KITE/data/initial_conditions_2022.rds"
out <- if (length(args) >= 2) args[2] else "dev/data/2022"

ic <- readRDS(path.expand(rds))
dir.create(out, recursive = TRUE, showWarnings = FALSE)

# `elasticities` is a nested list in the R object; lift it to the top level.
if (!is.null(ic$elasticities)) {
    for (nm in names(ic$elasticities)) ic[[nm]] <- ic$elasticities[[nm]]
    ic$elasticities <- NULL
}

# Expected index columns per table, and the full-grid size we compare against.
schema <- list(
    trade_share        = c("origin", "destination", "sector"),
    intermediate_share = c("country", "input", "output"),
    factor_share       = c("country", "sector"),
    consumption_share  = c("country", "sector"),
    expenditure        = c("country", "sector"),
    value_added        = c("country"),
    trade_balance      = c("country"),
    tariff             = c("origin", "destination", "sector"),
    export_subsidy     = c("origin", "destination", "sector"),
    trade_elasticity   = c("sector")
)

cat(sprintf("%-20s %10s %10s %10s\n", "table", "rows", "full grid", "absent"))
cat(strrep("-", 54), "\n")

for (nm in names(schema)) {
    x <- ic[[nm]]
    if (is.null(x)) next
    x <- as.data.table(x)
    idx <- schema[[nm]]
    if (!all(idx %in% names(x))) {
        cat(sprintf("%-20s SKIPPED (columns: %s)\n", nm, paste(names(x), collapse = ", ")))
        next
    }
    setcolorder(x, c(idx, "value"))
    grid <- prod(vapply(idx, function(k) uniqueN(x[[k]]), numeric(1)))
    cat(sprintf("%-20s %10d %10.0f %10.0f%s\n", nm, nrow(x), grid, grid - nrow(x),
                if (nrow(x) < grid) "  <- sparse" else ""))
    fwrite(x, file.path(out, paste0(nm, ".csv")))
}

cat("\nWritten to ", normalizePath(out), "\n", sep = "")
cat("Absent cells are filled by KITE.jl on load: 0 for shares and expenditure, ",
    "1 for tariffs and export subsidies.\n", sep = "")
