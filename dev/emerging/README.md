# KITE baseline from EMERGING V2

Builds a KITE initial-conditions database from the **EMERGING V2** multi-regional
input–output tables — 245 economies × 133 sectors, benchmark year 2023 — combined with
CEPII MAcMap-HS6 tariffs and Fontagné–Guimbard–Orefice product-level trade elasticities.

The result is **196 regions × 133 sectors**: every one of the 195 UN-recognised economies
kept separate, everything else pooled into `ROW`.

|  | shipped 2022 baseline (OECD ICIO) | this one (EMERGING V2) |
|---|---|---|
| regions | 81 | **196** |
| sectors | 50 | **133** |
| base year | 2022 | **2023** |
| trade-share cells | 328 050 | **5 109 328** |
| sector concordance for tariffs | HS6 → SITC3 → ICIO, hand-built | **HS6 → HS chapter, exact** |

## Why EMERGING makes the data problem easier

EMERGING's goods sectors *are* HS-2002 chapters: 96 of them (chapters 1–97 less 27, which
is disaggregated, and 77, which the nomenclature does not use), plus 7 energy/utility
sectors carved out of chapter 27, plus 30 EBOPS services. Both external datasets are
HS-based, so the concordance is an identity rather than a judgement call. The 2022 ICIO
build needed a hand-written 44-entry SITC-to-industry mapping with 3-digit overrides; here
every HS6 line belongs to exactly one sector by construction.

## Prerequisites

| Path | Description |
|---|---|
| `~/Documents/Data/EMERGING/V2/EMERGING_V2_2023_m.mat` | EMERGING V2 MRIO ([Zenodo 10.5281/zenodo.19461860](https://doi.org/10.5281/zenodo.19461860), CC BY 4.0) |
| `~/Documents/Data/EMERGING/classification/*.xlsx` | curated country and sector classifications |
| `~/Documents/R/KITE/data/raw/fontagne_elasticities/elast_hs6.dta` | HS6 trade elasticities |
| `~/Documents/R/KITE/data/raw/macmap_tariffs/Tariffs_2001_2019.zip` | MAcMap-HS6 bilateral applied tariffs |

R packages: `data.table`, `collapse`, `readxl`, `haven`, `rhdf5`, `africamonitor`.
`p7zip` must be on the path (`brew install p7zip`) — the MAcMap archive is ZIP64.

## Running

From the KITE.jl repo root, in order:

```bash
Rscript dev/emerging/01_extract_emerging.R
Rscript dev/emerging/02_elasticities.R
Rscript dev/emerging/03_tariffs.R
julia --project=. dev/emerging/04_load_and_test.jl
```

Everything lands in `dev/emerging/build/2023/` as the long CSVs `read_baseline_csv`
expects. Set `YEAR` in `00_common.R` to build 2015, 2018 or 2021 instead.

### `01_extract_emerging.R` (~4 min)

The transaction matrix is 32 585 × 32 585 — 8.5 GB dense. It is never held in memory. The
HDF5 chunk layout stores one column segment per chunk, so reading the 133 consecutive
columns that make up everything one destination economy buys is chunk-aligned; each block
is reduced on arrival and the 245 economies are pooled to the 196 regions on the fly. Peak
memory stays under 1 GB.

Two choices worth knowing about:

- **β uses supply-side gross output** (`rowSums(z) + rowSums(f)`) rather than the reported
  `X` vector. The two agree to 5 × 10⁻⁸ worldwide but disagree cell by cell, and only the
  supply-side figure is consistent with the π and X built alongside it.
- **Sectors that buy no intermediates** get their (near-zero weight) input bundle pointed
  at themselves, so the Cobb–Douglas input-cost aggregator stays a proper weighted mean
  instead of collapsing to an empty product.

### `02_elasticities.R`

θ = |ε|, the median over the HS6 lines in each chapter. Chapter 27 is split by 4-digit
heading to feed the energy sectors: coal 2701–04, crude oil 2709, gas 2711, refining
2706–08 + 2710 + 2712–15, manufactured gas 2705, electricity 2716.

The 30 EBOPS services and water take a flat **θ = 5**. No tariff-based method can identify
a services elasticity; Ahmad & Schreiber (USITC 2024) put the median Armington elasticity
of substitution for services at 5.98 across NAICS-3 industries (5.42 at GTAP-sector level,
5.01 for core tradable services), and the trade elasticity is σ − 1. Change
`SERVICES_THETA` to vary it.

`trade_elasticity_detail.csv` records, per sector, how many HS6 lines back the estimate and
which rule produced it. Two caveats live there rather than being silently smoothed away:

- **GAS (θ = 64.2)** is the only sector above the range Caliendo & Parro themselves use
  (0.37–51.08). It rests on 6 HS6 lines, 4 of them Fontagné et al.'s imputations for
  insignificant estimates.
- **ELY, GASD and OIL rest on a single HS6 line each.** That is not thin evidence so much
  as a homogeneous product — 2709 *is* crude petroleum — but it is worth knowing.

Estimates are reported as derived. Nothing is winsorised.

### `03_tariffs.R` (~4 min on a cold cache)

The 3.6 GB MAcMap CSV is streamed through `awk` and reduced to chapter means in a single
pass, never loaded. Within a cell the mean over HS6 lines is unweighted — the 2019 release
carries no trade values — but cells are 96 chapters rather than 26 industries, so far less
heterogeneity is averaged over than in the ICIO build.

151 of the 196 regions file their own import schedule. The rest are filled in two tiers,
and every cell in `tariff_2023_detail.csv` carries the tier it came from:

1. **customs union** — exact, since members apply a common external tariff. The EU
   (BEL, BGR, LUX, ROU, plus AND, MCO and SMR, which apply EU or EU-aligned tariffs), SACU
   (BWA, LSO, NAM, SWZ ← ZAF), and Switzerland–Liechtenstein. Applied on both the importer
   and the exporter side, since union members also receive common preferential access.
2. **regional median** over reporting economies in the same UN region — 33 importers and
   8 exporters, of which **TWN** is the only large trader.

Services and HS98 ("commodities not specified according to kind", which has no MAcMap
chapter) carry no tariff, as does self-trade.

`ROW`'s import schedule is the average of the free ports within it that do report
(Bermuda, Cayman Islands, Hong Kong), which is not representative of all 50 pooled
territories but is the only evidence available for them.

## Results

Calibration is exact and the null scenario — the correctness gate — is unity to machine
precision in a single iteration, as it is on the 81 × 50 baseline:

```
loaded and calibrated in 6.8 s      81 balancing passes
residuals: goods market 0.0e+00 | expenditure 3.4e-16 | income 0.0e+00
world value added 106 795 674 mn USD | Σ D = -1.1e-09
null scenario: 1 iteration, criterion 1.13e-16, 0.04 s
               max |ŵ-1| 2.2e-16, max |P̂-1| 2.2e-16, max |π'-π| 4.4e-16
```

Two counterfactuals, both at `vfactor = 0.05`:

**United States raises tariffs on China by 25 pp** — 71 iterations, 26 s. China loses
0.181% of welfare and the United States 0.098%; China's share of the US market falls to
0.23 of its baseline in the median traded sector. The gainers are the economies trade
diversion actually favours — Cambodia +0.39%, Vietnam +0.29%, Mexico +0.15% — and the
collateral losers are China's own upstream suppliers: Turkmenistan (gas) −0.29%, Mongolia
(coal) −0.24%, Laos −0.09%.

**AfCFTA, tariffs eliminated among the 54 African members** — 57 iterations, 6.7 s. Mean
welfare rises 0.074% across members against −0.0003% for the rest of the world, with the
gains concentrated in high-tariff transit economies: Djibouti +2.56%, Togo +1.95%,
Eswatini +0.43%, Uganda +0.36%. Tariff-only magnitudes of this order are what the AfCFTA
literature reports; the large estimated gains come from non-tariff measures, which this
scenario does not touch.

`ChowdhryHinzKaminWanner2022` with an empty coalition reproduces `CaliendoParro2015` to
8.9 × 10⁻⁶, i.e. to solver tolerance.

## Solver settings: use `vfactor = 0.05`

**The package default `vfactor = 0.2` does not converge at 196 regions.** It is above the
stability limit of the wage iteration at this dimension, and the failure is not a
large-shock effect — a 1 pp tariff diverges exactly as a 25 pp one does. `04c_stability.jl`
separates the candidates:

| setting | outer iterations | criterion | converged | wall |
|---|---|---|---|---|
| default `vfactor = 0.2` | 1000 (cap) | 1.3 × 10⁻² | **no** | 394 s |
| `vfactor = 0.10` | 300 (cap) | 6.0 × 10⁻³ | **no** | 68 s |
| **`vfactor = 0.05`** | **71** | **9.9 × 10⁻⁷** | **yes** | **25 s** |
| `vfactor = 0.02` | 146 | 9.9 × 10⁻⁷ | yes | 44 s |
| θ capped at 30, `vfactor = 0.2` | 300 (cap) | 2.7 × 10⁻² | no | 96 s |
| θ capped at 20, `vfactor = 0.2` | 300 (cap) | 1.4 × 10⁻² | no | 65 s |
| θ capped at 15, `vfactor = 0.2` | 112 | 8.8 × 10⁻⁷ | yes | 17 s |

The elasticity distribution does play a part — an aggressive cap at 15 restores
convergence on its own — but a cap that low rewrites some 40 sectors' estimates, so
damping is the fix that leaves the data alone. Being stable, `vfactor = 0.05` is also by
far the fastest: 25 seconds against 394 wasted ones.

A non-converged run is not obviously wrong from its numbers alone. At `vfactor = 0.2` the
US–China counterfactual returned a 5% welfare *gain* for Mongolia and a 6% loss for Qatar;
at 0.05 those disappear. Always check `res.converged`.

## Known limitations

- **Tariffs are 2019, the IO table is 2023.** MAcMap-HS6 has no later release. Levels are
  therefore four years stale; this matters for the baseline tariff revenue term, not for
  the hat algebra, which responds only to changes.
- **The US–China Section 301 tariffs appear not to be in the data.** MAcMap 2019 puts the
  US applied rate on Chinese HS85 (electrical machinery) at 1.6% and HS72 (iron and steel)
  at 0.3%, which are MFN rates — the 2018–19 unilateral actions are not reflected. Treat
  the US–China baseline as pre-trade-war and set those tariffs explicitly in a scenario if
  you need them. Spot checks elsewhere are accurate: US apparel 12.0%, India vehicles
  35.2%, EU meat 27.0%, Japan cereals 47.8%.
- **Unweighted HS6 averages within a chapter** — see above.
- **Regionally imputed tariffs for 33 importers**, all small except Taiwan.
- **Flat services elasticity** across 30 sectors that certainly differ.
- Negative and undefined β cells (subsidies, MRIO balancing items) are clipped into
  (10⁻⁴, 1 − 10⁻⁴); the count is reported by `01_extract_emerging.R`.

## Citation

Huo, J., Wang, W., & Guan, D. (2026). *Multi-regional Input–output Table for the Global
Emerging Economies (EMERGING) V2 (2015–2023)* [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.19461860

Huo, J., Chen, P., Hubacek, K., Zheng, H., Meng, J., & Guan, D. (2022). Full-scale, near
real-time multi-regional input–output table for the global emerging economies (EMERGING).
*Journal of Industrial Ecology*, 26, 1218–1232.

Fontagné, L., Guimbard, H., & Orefice, G. (2022). Tariff-based product-level trade
elasticities. *Journal of International Economics*, 137, 103593.
