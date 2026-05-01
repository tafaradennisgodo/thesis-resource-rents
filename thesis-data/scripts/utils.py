"""Shared utilities for thesis data pulls."""

import logging
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
CLEAN_DIR = ROOT / "clean"
LOGS_DIR = ROOT / "logs"
README = ROOT / "README.md"

for d in (RAW_DIR, CLEAN_DIR, LOGS_DIR):
    d.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{name}_{ts}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def raw_path(name: str, ext: str = "csv") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RAW_DIR / f"{name}_{ts}.{ext}"


def summarise(df, label: str, logger: logging.Logger) -> None:
    logger.info(f"=== {label} summary ===")
    logger.info(f"  Rows:      {len(df):,}")
    if "iso3" in df.columns:
        logger.info(f"  Countries: {df['iso3'].nunique()}")
    if "year" in df.columns:
        logger.info(f"  Years:     {int(df['year'].min())}–{int(df['year'].max())}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.warning("  Missing values:")
        for col, n in missing.items():
            pct = 100 * n / len(df)
            logger.warning(f"    {col}: {n:,} ({pct:.1f}%)")
    else:
        logger.info("  No missing values.")


def append_readme(
    dataset: str,
    source_url: str,
    variables: dict[str, str],
    raw_file: Path,
    clean_file: Path,
    n_rows: int,
    n_countries: int,
    year_range: tuple[int, int],
) -> None:
    date_pulled = datetime.now().strftime("%Y-%m-%d")

    var_str = ", ".join(f"`{k}` ({v})" for k, v in variables.items())
    row = (
        f"| — | {dataset} | {source_url} | {var_str} | {date_pulled} "
        f"| {raw_file.name} | {clean_file.name} "
        f"| {n_rows:,} | {n_countries} | {year_range[0]}–{year_range[1]} |\n"
    )

    text = README.read_text()
    marker = "| # | Dataset |"
    # Insert after the header + separator rows of the table
    lines = text.splitlines(keepends=True)
    insert_at = None
    for i, line in enumerate(lines):
        if line.strip().startswith(marker.strip()):
            insert_at = i + 2  # skip header + separator
            break
    if insert_at is None:
        README.write_text(text + "\n" + row)
        return
    lines.insert(insert_at, row)
    README.write_text("".join(lines))

    # Variable definitions block
    defn_block = f"\n### {dataset}\n\n"
    for k, v in variables.items():
        defn_block += f"- **`{k}`**: {v}\n"
    defn_block += "\n"

    text2 = README.read_text()
    defn_marker = "## Variable Definitions"
    if defn_marker in text2:
        text2 = re.sub(
            rf"(## Variable Definitions\n)",
            r"\1" + defn_block,
            text2,
            count=1,
        )
        README.write_text(text2)
