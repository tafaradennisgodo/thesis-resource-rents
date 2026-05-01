"""
Pull World Bank WDI data for the thesis panel dataset.

Outputs:
  clean/wdi_panel.csv      - long-format country-year panel
  clean/country_list.csv   - master country list with income group / region
  logs/wdi_pull_<ts>.log   - full run log
"""

import sys
from pathlib import Path

import time
import concurrent.futures

import pandas as pd
import wbgapi as wb

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    CLEAN_DIR,
    RAW_DIR,
    append_readme,
    get_logger,
    raw_path,
    summarise,
)

# ── Configuration ────────────────────────────────────────────────────────────

YEARS = range(1995, 2024)  # 1995–2023

SERIES = {
    "NY.GDP.MKTP.CD":       "GDP, current US$",
    "NY.GNP.MKTP.CD":       "GNI, current US$",
    "NY.GDP.PCAP.KD":       "GDP per capita, constant 2015 US$",
    "BX.GSR.NFCY.CD":       "Primary income receipts from abroad (BoP, current US$)",
    "BM.GSR.NFCY.CD":       "Primary income payments abroad (BoP, current US$)",
    "NY.GDP.TOTL.RT.ZS":    "Total natural resource rents, % GDP",
    "NY.GDP.PETR.RT.ZS":    "Oil rents, % GDP",
    "NY.GDP.MINR.RT.ZS":    "Mineral rents, % GDP",
    "NY.GDP.NGAS.RT.ZS":    "Natural gas rents, % GDP",
    "NY.GDP.COAL.RT.ZS":    "Coal rents, % GDP",
    "NY.GDP.FRST.RT.ZS":    "Forest rents, % GDP",
    "BX.KLT.DINV.WD.GD.ZS": "FDI net inflows, % GDP",
    "BM.KLT.DINV.WD.GD.ZS": "FDI net outflows, % GDP",
    "SP.POP.TOTL":           "Population",
    "NE.TRD.GNFS.ZS":       "Trade openness (exports + imports), % GDP",
    "FP.CPI.TOTL.ZG":       "Inflation, consumer prices (annual %)",
}

INCOME_GROUPS = {"LIC", "LMC", "UMC"}  # low, lower-middle, upper-middle income
SSA_REGION_CODE = "SSF"                 # World Bank Sub-Saharan Africa code

README_SOURCE = "https://databank.worldbank.org/source/world-development-indicators"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_id(field) -> str:
    """Extract an ID from a WB metadata field that may be a dict, str, or None."""
    if isinstance(field, dict):
        return field.get("id", "") or ""
    if isinstance(field, str):
        return field
    return ""


def fetch_country_metadata(log) -> pd.DataFrame:
    """Return DataFrame of all WB countries with iso3, name, income group, region."""
    log.info("Fetching country metadata from WB API …")
    records = []
    first = True
    for c in wb.economy.list():
        if first:
            il = c.get("incomeLevel")
            rg = c.get("region")
            log.info(f"  DEBUG first economy '{c['id']}': "
                     f"incomeLevel type={type(il).__name__!r} value={il!r}  |  "
                     f"region type={type(rg).__name__!r} value={rg!r}")
            first = False
        records.append({
            "iso3":         c["id"],
            "country":      c["value"],
            "income_group": _extract_id(c.get("incomeLevel")),
            "region":       _extract_id(c.get("region")),
        })
    df = pd.DataFrame(records)
    log.info(f"  Total economies returned: {len(df)}")
    return df


def build_sample_frame(meta: pd.DataFrame, log) -> tuple[pd.DataFrame, list[str]]:
    """Filter to LIC/LMC/UMC countries; flag aggregates and unclassified entries."""
    # Entries with blank or 'NA' income group are aggregates / not classifiable
    aggregates = meta[~meta["income_group"].isin(INCOME_GROUPS) | (meta["income_group"] == "")]
    if not aggregates.empty:
        log.warning(
            f"  {len(aggregates)} entries NOT in a LIC/LMC/UMC income group "
            "(likely aggregates or high-income — excluded):"
        )
        for _, r in aggregates.iterrows():
            log.warning(f"    {r['iso3']:6s}  {r['income_group']:6s}  {r['country']}")

    sample = meta[meta["income_group"].isin(INCOME_GROUPS)].copy()
    sample["ssa_dummy"] = (sample["region"] == SSA_REGION_CODE).astype(int)
    iso3_list = sample["iso3"].tolist()
    log.info(
        f"  Sample frame: {len(iso3_list)} countries "
        f"({sample['ssa_dummy'].sum()} SSA)"
    )
    return sample, iso3_list


FETCH_TIMEOUT   = 60   # seconds before falling back to chunked pull
CHUNK_SIZE      = 5    # series per chunk in fallback mode
RATE_LIMIT_WAIT = 10   # seconds to wait after a 429 before retry


def _wb_dataframe(series_codes: list[str], iso3_list: list[str]) -> pd.DataFrame:
    """Single wbgapi call — runs in a thread so it can be interrupted."""
    return wb.data.DataFrame(
        series_codes,
        economy=iso3_list,
        time=YEARS,
        skipBlanks=False,
        skipAggs=True,
        numericTimeKeys=True,
    )


def _raw_to_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert wbgapi wide output (MultiIndex × year columns) to long format."""
    raw = raw.reset_index()
    raw.columns.name = None
    if "economy" not in raw.columns and "Country" in raw.columns:
        raw = raw.rename(columns={"Country": "economy"})
    if "series" not in raw.columns and "Series" in raw.columns:
        raw = raw.rename(columns={"Series": "series"})

    id_vars   = [c for c in raw.columns if c in {"economy", "series"}]
    year_cols = [c for c in raw.columns if str(c).isdigit()]

    long = raw.melt(id_vars=id_vars, value_vars=year_cols,
                    var_name="year", value_name="value")
    long["year"]  = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.rename(columns={"economy": "iso3"})


def _fetch_with_retry(series_codes: list[str], iso3_list: list[str],
                      log, label: str = "batch") -> pd.DataFrame:
    """Call _wb_dataframe once; retry once on HTTP 429."""
    for attempt in (1, 2):
        try:
            return _wb_dataframe(series_codes, iso3_list)
        except wb.APIError as exc:
            if attempt == 1 and "429" in str(exc):
                log.warning(f"  [{label}] Rate-limited (429) — waiting {RATE_LIMIT_WAIT}s …")
                time.sleep(RATE_LIMIT_WAIT)
            else:
                raise


def _chunked_pull(series_codes: list[str], iso3_list: list[str], log) -> pd.DataFrame:
    """Fallback: pull CHUNK_SIZE series at a time and concatenate."""
    chunks   = [series_codes[i:i + CHUNK_SIZE]
                for i in range(0, len(series_codes), CHUNK_SIZE)]
    log.warning(f"  Falling back to chunked pull: {len(chunks)} chunks "
                f"of ≤{CHUNK_SIZE} series each.")
    longs = []
    for idx, chunk in enumerate(chunks, 1):
        label = f"chunk {idx}/{len(chunks)} {chunk}"
        log.info(f"  Fetching {label} …")
        t0  = time.time()
        raw = _fetch_with_retry(chunk, iso3_list, log, label=label)
        longs.append(_raw_to_long(raw))
        log.info(f"  Done {label} — {len(longs[-1]):,} rows in {time.time()-t0:.1f}s")
    return pd.concat(longs, ignore_index=True)


def pull_series(iso3_list: list[str], log) -> pd.DataFrame:
    """Pull all SERIES for the sample frame and return a long-format DataFrame."""
    series_codes = list(SERIES.keys())
    log.info(f"Pulling {len(series_codes)} series for {len(iso3_list)} countries, "
             f"years {min(YEARS)}–{max(YEARS)} …")

    # Attempt 1: single batched call with a hard timeout.
    # wbgapi has no native timeout, so we run it in a thread and cancel if slow.
    log.info(f"  Fetching all series in one batch (timeout={FETCH_TIMEOUT}s) …")
    t0 = time.time()
    long: pd.DataFrame | None = None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_fetch_with_retry, series_codes, iso3_list, log, "full batch")
        try:
            raw  = future.result(timeout=FETCH_TIMEOUT)
            long = _raw_to_long(raw)
            log.info(f"  Fetched {len(long):,} rows in {time.time()-t0:.1f}s")
        except concurrent.futures.TimeoutError:
            log.warning(f"  Single-batch call exceeded {FETCH_TIMEOUT}s — switching to chunked pull.")
            future.cancel()

    if long is None:
        t0   = time.time()
        long = _chunked_pull(series_codes, iso3_list, log)
        log.info(f"  Chunked pull complete: {len(long):,} rows in {time.time()-t0:.1f}s")

    return long


def pivot_to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot series column to wide format."""
    wide = long.pivot_table(
        index=["iso3", "year"],
        columns="series",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def add_country_names(wide: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Merge in country name from metadata."""
    names = meta[["iso3", "country"]].drop_duplicates()
    return wide.merge(names, on="iso3", how="left")


def compute_derived(df: pd.DataFrame, log) -> pd.DataFrame:
    """Add constructed variables."""
    gdp = df["NY.GDP.MKTP.CD"]
    gni = df["NY.GNP.MKTP.CD"]
    bx  = df["BX.GSR.NFCY.CD"]
    bm  = df["BM.GSR.NFCY.CD"]

    df["income_leak"] = (gdp - gni) / gdp
    df["net_primary_income_pct_gdp"] = (bx - bm) / gdp

    n_leak = df["income_leak"].notna().sum()
    n_npi  = df["net_primary_income_pct_gdp"].notna().sum()
    log.info(f"  income_leak non-missing:              {n_leak:,} / {len(df):,}")
    log.info(f"  net_primary_income_pct_gdp non-missing: {n_npi:,} / {len(df):,}")
    return df


def report_summary_stats(df: pd.DataFrame, log) -> None:
    """Print mean/sd/min/max/N/pct-missing for key variables."""
    key_vars = ["income_leak", "NY.GDP.TOTL.RT.ZS", "BX.KLT.DINV.WD.GD.ZS"]
    log.info("─" * 60)
    log.info("Summary statistics (key variables):")
    for v in key_vars:
        if v not in df.columns:
            log.warning(f"  {v}: column not found")
            continue
        s = df[v]
        n_total   = len(s)
        n_valid   = s.notna().sum()
        pct_miss  = 100 * (1 - n_valid / n_total)
        log.info(
            f"  {v}:\n"
            f"    N={n_valid:,}  missing={pct_miss:.1f}%\n"
            f"    mean={s.mean():.4f}  sd={s.std():.4f}  "
            f"min={s.min():.4f}  max={s.max():.4f}"
        )
    log.info("─" * 60)


def report_missing_countries(df: pd.DataFrame, log) -> None:
    """Flag countries with >50% of years missing for income_leak and resource rents."""
    n_years = df["year"].nunique()
    threshold = 0.50

    for var, label in [
        ("income_leak",       "income_leak"),
        ("NY.GDP.TOTL.RT.ZS", "total resource rents"),
    ]:
        if var not in df.columns:
            continue
        miss = (
            df.groupby("iso3")[var]
            .apply(lambda s: s.isna().sum() / n_years)
            .reset_index(name="miss_frac")
        )
        bad = miss[miss["miss_frac"] > threshold].sort_values("miss_frac", ascending=False)
        if bad.empty:
            log.info(f"  No countries >50% missing for {label}.")
        else:
            log.warning(f"  Countries with >50% years missing for {label} ({len(bad)}):")
            for _, r in bad.iterrows():
                log.warning(f"    {r['iso3']:6s}  {r['miss_frac']*100:.0f}% missing")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log = get_logger("wdi_pull")
    log.info("=== WDI pull started ===")

    # 1. Country metadata
    meta = fetch_country_metadata(log)
    sample_meta, iso3_list = build_sample_frame(meta, log)

    # 2. Pull series
    long = pull_series(iso3_list, log)

    # 3. Save raw
    raw_file = raw_path("wdi_raw", "csv")
    long.to_csv(raw_file, index=False)
    log.info(f"Raw file saved: {raw_file.name}  ({len(long):,} rows)")

    # 4. Pivot to wide
    wide = pivot_to_wide(long)
    wide = add_country_names(wide, sample_meta)

    # 5. Merge in ssa_dummy
    wide = wide.merge(
        sample_meta[["iso3", "ssa_dummy"]],
        on="iso3", how="left"
    )

    # 6. Computed variables
    wide = compute_derived(wide, log)

    # 7. Column ordering
    id_cols   = ["iso3", "country", "year", "ssa_dummy"]
    data_cols = [c for c in wide.columns if c not in id_cols]
    wide = wide[id_cols + data_cols]
    wide = wide.sort_values(["iso3", "year"]).reset_index(drop=True)

    # 8. Summary
    summarise(wide, "wdi_panel", log)
    report_summary_stats(wide, log)
    report_missing_countries(wide, log)

    # 9. Save clean panel
    clean_panel = CLEAN_DIR / "wdi_panel.csv"
    wide.to_csv(clean_panel, index=False)
    log.info(f"Clean panel saved: {clean_panel}  ({len(wide):,} rows)")

    # 10. Save country list
    country_list = sample_meta[["iso3", "country", "income_group", "region", "ssa_dummy"]].copy()
    country_list = country_list.sort_values("iso3").reset_index(drop=True)
    clean_countries = CLEAN_DIR / "country_list.csv"
    country_list.to_csv(clean_countries, index=False)
    log.info(f"Country list saved: {clean_countries}  ({len(country_list)} countries)")

    # 11. README
    all_vars = {**SERIES,
                "income_leak":               "Constructed: (GDP - GNI) / GDP",
                "net_primary_income_pct_gdp": "Constructed: (BX.GSR.NFCY.CD - BM.GSR.NFCY.CD) / NY.GDP.MKTP.CD",
                "ssa_dummy":                 "1 if World Bank Sub-Saharan Africa region (SSF), else 0"}
    append_readme(
        dataset="WDI Panel",
        source_url=README_SOURCE,
        variables=all_vars,
        raw_file=raw_file,
        clean_file=clean_panel,
        n_rows=len(wide),
        n_countries=wide["iso3"].nunique(),
        year_range=(int(wide["year"].min()), int(wide["year"].max())),
    )
    log.info("README.md updated.")
    log.info("=== WDI pull complete ===")


if __name__ == "__main__":
    main()
