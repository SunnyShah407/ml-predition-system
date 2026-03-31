from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlps.data.synthetic_tender import SyntheticTenderConfig, generate_synthetic_tenders  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate realistic synthetic tender dataset (CSV).")
    p.add_argument(
        "--config",
        default="configs/synthetic_tender.yaml",
        help="Path to YAML config (default: configs/synthetic_tender.yaml)",
    )
    p.add_argument("--output", default=None, help="Override output CSV path")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    ds = cfg_raw.get("dataset", {}) if isinstance(cfg_raw, dict) else {}
    output_csv = args.output or ds.get("output_csv", "data/raw/synthetic_tenders.csv")

    cfg = SyntheticTenderConfig(
        n_tenders=int(ds.get("n_tenders", 8000)),
        n_buyers=int(ds.get("n_buyers", 120)),
        n_vendors=int(ds.get("n_vendors", 450)),
        start_date=str(ds.get("start_date", "2023-01-01")),
        end_date=str(ds.get("end_date", "2025-12-31")),
        seed=int(ds.get("seed", 42)),
    )

    df = generate_synthetic_tenders(cfg)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df):,} rows to {out_path}")
    print("Target column: winning_total_price")


if __name__ == "__main__":
    main()

