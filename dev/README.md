# Development scripts and baseline data

Not part of the package. These build the baselines and cross-validate the solver against the
R implementation. Both need R with `data.table`, and a checkout of the R package.

Two baselines are buildable here. Neither is committed — `dev/data/`, `dev/emerging/build/`
and `data/kite_baseline_*/` are all gitignored, because the outputs run from 9 MB to 450 MB.

| | ICIO 2022 | EMERGING 2023 |
|---|---|---|
| build scripts | `export_baseline_from_R.R`, `build_baseline_artifact.jl` | [`emerging/`](emerging/) |
| output | `dev/data/2022/`, `data/kite_baseline_2022/` | `dev/emerging/build/2023/` |
| regions × sectors | 81 × 50 | 196 × 133 |
| base year | 2022 | 2023 |
| MRIO source | OECD ICIO 2025 SML | EMERGING V2 |
| trade-share cells | 328 050 | 5 109 328 |
| loader | `load_baseline(year = 2022)` | `read_baseline_csv("dev/emerging/build/2023")` |

---

# 1. The data

## 1.1 Units and conventions

- **All levels are millions of current US dollars**, inherited from the source MRIO. The
  model is homogeneous of degree one in nominal income, so the unit only sets the scale of
  reported levels.
- **All shares are fractions**, not percentages.
- **Tariffs and export subsidies are multipliers**: `1.0` is free trade, `1.25` a 25%
  ad-valorem tariff. Never rates.
- **Trade tensors are indexed `[origin, destination, sector]`**, so a fixed sector is a
  contiguous `N × N` matrix in Julia's column-major layout.
- **`D` is the trade *surplus*** and enters income with a minus sign. A country running a
  deficit has `D < 0`.
- Country codes are ISO 3166-1 alpha-3. Sector codes differ by baseline (§1.4, §1.5).

## 1.2 The long CSV interchange format

The documented, portable format, and the same schema the R package uses — an existing KITE
dataset transfers unchanged. One file per variable, each with its index columns plus
`value`. `read_baseline_csv(dir)` reads a directory of them and calibrates.

| file | index columns | definition | normalisation |
|---|---|---|---|
| `trade_share.csv` | `origin, destination, sector` | π: share of destination's expenditure on that sector sourced from origin | Σ over `origin` = 1 |
| `intermediate_share.csv` | `country, input, output` | γ: share of `input` in the intermediate cost of `output` | Σ over `input` = 1 |
| `factor_share.csv` | `country, sector` | β: value added ÷ gross output | in (0, 1) |
| `expenditure.csv` | `country, sector` | X: total absorption, intermediate + final | level |
| `trade_elasticity.csv` | `sector` | θ: Fréchet dispersion, **standard convention** — larger θ means more responsive trade | > 0 |
| `tariff.csv` *(optional)* | `origin, destination, sector` | τ multiplier | ≥ 1 typically |
| `export_subsidy.csv` *(optional)* | `origin, destination, sector` | ζ multiplier | |
| `consumption_share.csv` *(optional)* | `country, sector` | α: final-absorption shares | Σ over `sector` = 1 |
| `value_added.csv` *(optional)* | `country` | VA | level |
| `trade_balance.csv` *(optional)* | `country` | D, trade surplus | level, sums to ≈ 0 |

The first five are required. **Tables may be sparse**: absent cells take a documented fill —
`0` for shares and expenditure, `1` for tariffs and export subsidies — and the loader reports
how many it filled. This is why `trade_share.csv` holds 1.78 M rows rather than 5.11 M.

Two consequences worth internalising. Rows are matched **by label, not by position**, so a
row order or an incomplete grid cannot silently corrupt the arrays; the R implementation
reshapes positionally and does corrupt them on sparse input. And θ is the **standard**
convention here — the R package's parameter is its reciprocal (see `NEWS.md`), so a θ
carried over from R must be inverted.

### What calibration does to it

`read_baseline_csv` does not use the data as given. The equilibrium identities over-determine
raw MRIO data, so `calibrate` holds the behavioural shares (π, γ, β, θ) and observed
expenditure X fixed and lets final demand and the trade balance absorb the inconsistency:
Y and intermediate demand are recomputed, residual final demand `F = X − ID` is clipped at
zero and X reset until `F ≥ 0` everywhere, α is derived as `F / Σ F`, and
`D = VA + R − I` closes the income identity. **Any `consumption_share.csv`,
`value_added.csv` or `trade_balance.csv` you supply is therefore overwritten** under the
default `final_demand = :residual`; they are shipped for reference and for
`anchor = :value_added`. On the EMERGING build the loop takes 81 passes and lands on goods
market 0.0, expenditure 3.4 × 10⁻¹⁶, income 0.0.

## 1.3 The dense binary format

What the shipped artifact holds, because parsing 830 k CSV rows costs about half a second
while `read!` costs ten milliseconds. Written by `write_baseline_binary`, read by
`read_baseline_binary`; the levels are already model-consistent, so no calibration runs.

```
kite_baseline_2022/
├── meta.toml        [dataset] year, source, layout   [dims] N, J
├── countries.txt    N lines, one code each, in array order
├── sectors.txt      J lines
└── *.bin            column-major little-endian Float64, no header
```

Blob shapes follow from `N` and `J`: `(N,N,J)` for π, τ, ζ; `(N,J,J)` for γ; `(N,J)` for
α, β, X, Y; `(J,)` for θ; `(N,)` for I, R, VA, D. There is no magic number and no checksum
in the file itself — integrity comes from the artifact SHA.

## 1.4 ICIO 2022 baseline — 81 × 50

81 economies (80 ISO plus ROW) × 50 OECD ICIO 2025 SML sectors, ISIC Rev. 4 aggregates with
codes like `A01`, `C10T12`. Tariffs are MAcMap-HS6 2019 concorded HS6 → SITC3 → ICIO;
elasticities are Fontagné et al. aggregated to ICIO sectors, range 1.4–14.8.

Built from `~/Documents/R/KITE/data/initial_conditions_2022.rds`; that repo's
`misc/initial_conditions_format.md` and `data/README.md` document the upstream pipeline.

## 1.5 EMERGING 2023 baseline — 196 × 133

Full provenance, method and limitations are in [`emerging/README.md`](emerging/README.md).
The data itself:

**196 regions.** All 195 UN-recognised economies (`africamonitor::am_countries_wld`) kept
separate, plus `ROW`. `ROW` pools the 50 remaining EMERGING territories — of which 8 (ANT,
GLP, GUF, MTQ, MYT, REU, SJM, VIR) are entirely empty in the source, and the largest real
ones are **Hong Kong and Macao**. Neither is in the UN list, so anyone doing East Asian work
should know they are inside ROW rather than separate. Puerto Rico and Palestine are there
too.

**133 sectors**, in three groups:

| group | n | codes | θ range |
|---|---|---|---|
| HS-2002 chapters | 96 | `HS01`–`HS98` (chapters 1–97 less 27 and the unused 77, plus 98 "not specified") | 2.2 – 32.8 |
| energy and utilities, carved out of HS chapter 27 | 7 | `COAL` `OIL` `GAS` `PETR` `ELY` `GASD` `WATR` | 5.0 – 64.2 |
| EBOPS services | 30 | `MSPI` `MAIN` `TSEA` `TAIR` `TOTH` `POST` `TRVG` `TLOC` `ACCO` `FOOD` `CONS` `INSD` `PENS` `FINS` `REAL` `IPCH` `TELE` `COMP` `INFO` `RSDV` `PROF` `ENGI` `WAST` `LEAS` `OBUS` `AUDV` `HLTH` `EDUC` `RECR` `GOVT` | 5.0 flat |

`emerging/build/2023/trade_elasticity_detail.csv` carries, per sector, the HS chapter it came
from, how many HS6 lines back the estimate, and which rule produced it — read it before
trusting any single θ. Services have no tariff, and neither does `WATR` or `HS98`.

### Files produced

```
dev/emerging/build/
├── macmap_hs2_2019.csv        53 MB   awk reduction of the 3.6 GB MAcMap file, cached
├── macmap_hs4_ch27_2019.csv  8.5 MB   chapter 27 at 4-digit, for the energy sectors
├── tariff_2023_detail.csv     214 MB  every tariff cell with its provenance tier
└── 2023/
    ├── *.csv                  237 MB  the long interchange format of §1.2
    ├── trade_elasticity_detail.csv    per-sector θ provenance
    ├── arrays.rds              39 MB  R-side arrays before the CSV round-trip
    └── baseline.jls           179 MB  the *calibrated* KiteBaseline, serialized
```

`baseline.jls` is the one to load for analysis — `deserialize` is instant against 6.8 s of
CSV parsing plus calibration. It is a Julia serialization, so it is **not portable across
Julia versions or machines**; rebuild it with `04a_load.jl` rather than copying it.

### Solver setting

**Use `vfactor = 0.05`.** The package default of `0.2` is above the stability limit of the
wage iteration at 196 regions and does not converge — for a 1 pp tariff just as for 25 pp.
The evidence and the alternatives are tabulated in [`emerging/README.md`](emerging/README.md)
and reproducible with `emerging/04c_stability.jl`. A non-converged run does not look wrong
from its numbers alone, so check `res.converged`.

---

# 2. Building

## 2.1 ICIO 2022

```bash
# 1. RDS -> long CSVs. Reports which tables are sparse.
Rscript dev/export_baseline_from_R.R ~/Documents/R/KITE/data/initial_conditions_2022.rds dev/data/2022

# 2. CSVs -> calibrated dense binary in data/kite_baseline_2022/
julia --project=. dev/build_baseline_artifact.jl 2022
```

Step 2 refuses to write unless the calibrated baseline reproduces itself under a no-change
scenario. `load_baseline(year = 2022)` picks the resulting directory up directly.

## 2.2 EMERGING 2023

```bash
Rscript dev/emerging/01_extract_emerging.R    # ~4 min, streams the 8.5 GB MRIO
Rscript dev/emerging/02_elasticities.R
Rscript dev/emerging/03_tariffs.R             # ~4 min on a cold cache
julia --project=. dev/emerging/04a_load.jl    # calibrate + null-scenario gate
```

Set `YEAR` in `emerging/00_common.R` to build 2015, 2018 or 2021 instead. Note that both R
scripts resolve their inputs through hardcoded absolute paths in `00_common.R`, so they are
reproducible on this machine but not on a fresh checkout without the raw sources.

## 2.3 Publishing a baseline as an artifact

The 2022 baseline is ~9.2 MB, too large to commit. To publish:

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

The EMERGING baseline has not been published this way. Its dense binary would be ~250 MB
before compression, so it wants its own release asset and its own `Artifacts.toml` entry
rather than riding along with the 2022 one.

---

# 3. Cross-validating against R

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
