# Development scripts

Not part of the package. These build the shipped baseline and cross-validate the solver against
the R implementation. Both need R with `data.table`, and a checkout of the R package.

## Building a baseline

```bash
# 1. RDS -> long CSVs. Reports which tables are sparse.
Rscript dev/export_baseline_from_R.R ~/Documents/R/KITE/data/initial_conditions_2022.rds dev/data/2022

# 2. CSVs -> calibrated dense binary in data/kite_baseline_2022/
julia --project=. dev/build_baseline_artifact.jl 2022
```

Step 2 refuses to write unless the calibrated baseline reproduces itself under a no-change
scenario. `load_baseline(year = 2022)` picks the resulting directory up directly, so the
baseline is usable immediately; `data/` is gitignored.

### Publishing as an artifact

The baseline is ~9.2 MB, too large to commit. To publish:

```bash
cd data && tar czf kite_baseline_2022.tar.gz kite_baseline_2022 && shasum -a 256 kite_baseline_2022.tar.gz
```

Upload the tarball to a GitHub release, then add to `Artifacts.toml`:

```toml
[kite_baseline_2022]
git-tree-sha1 = "<output of `julia -e 'using Tar, Inflate, SHA; println(Tar.tree_hash(IOBuffer(inflate_gzip("kite_baseline_2022.tar.gz"))))'`>"
lazy = true

    [[kite_baseline_2022.download]]
    url = "https://github.com/SebKrantz/KITE.jl/releases/download/v0.1.0/kite_baseline_2022.tar.gz"
    sha256 = "<shasum output>"
```

`load_baseline` resolves in this order: `KITE_BASELINE_DIR`, then `data/kite_baseline_<year>/`,
then the artifact.

## Cross-validating against R

```bash
Rscript dev/validate_against_R.R ~/Documents/R/KITE test/fixtures/toy_3x2 test/fixtures/golden_cp2015_3x2.csv
julia --project=. -e 'using Pkg; Pkg.test()'
```

`validate_against_R.R` sources the R package and overrides
`update_price_index_cp_2015` / `update_trade_share_cp_2015` to restore the published exponent
convention, then feeds it complete Cartesian grids so its positional array casting cannot
corrupt them. Everything else is left as the R package has it, so agreement is genuine evidence
for both implementations. It writes a golden CSV that the Julia test suite asserts against at
`rtol = 1e-9`; the two currently agree to `1e-15`.

Regenerate the golden file whenever the fixture or the solver's equations change.

The full 81 × 50 comparison is not part of CI — it needs the built baseline and an R install.
To run it manually, point `validate_against_R.R` at `dev/data/2022` instead of the fixture.
