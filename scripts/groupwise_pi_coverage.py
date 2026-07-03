"""
groupwise_pi_coverage.py
========================
Compute empirical 95% PI coverage separately for MDD and HC subjects from a
train_svr.py results dir. Augments the marginal PI coverage already in
svr_run_results.csv with group-conditional coverage: the diagnostic that
catches systematic miscalibration.

Inputs:
  --results-dir   path to a train_svr.py timestamped results dir
                  (must contain svr_participant_results.csv + svr_run_results.csv)
  --metadata      path to subject_info_map.csv (subject_id + group columns)

Outputs:
  results-dir/pi_coverage_by_group.csv  — one row per run, all metrics
  Prints sorted-by-PI-width table to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def normalize_subject_id(s) -> int:
    s = str(s).lstrip("0")
    return int(s) if s else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path,
        default=Path("data/metadata/subject_info_map.csv"))
    args = parser.parse_args(argv)

    parts = pd.read_csv(args.results_dir / "svr_participant_results.csv")
    runs  = pd.read_csv(args.results_dir / "svr_run_results.csv")
    info  = pd.read_csv(args.metadata)
    info["subject_id"] = info["subject_id"].apply(normalize_subject_id)
    groups = info.set_index("subject_id")["group"]

    parts["group"] = parts["subject_id"].map(groups)
    if parts["group"].isna().any():
        n_miss = parts["group"].isna().sum()
        print(f"WARNING: {n_miss} rows have no group label", file=sys.stderr)

    by_group = (parts.groupby(["run", "group"])["in_interval"]
                .mean().unstack("group"))

    out = (runs[["run", "RMSE", "PI_coverage", "PI_mean_width"]]
           .merge(by_group.reset_index(), on="run", how="left")
           .sort_values("PI_mean_width")
           .rename(columns={"PI_coverage": "coverage_marginal",
                            "PI_mean_width": "PI_width",
                            "HC": "coverage_HC",
                            "MDD": "coverage_MDD"}))

    out_csv = args.results_dir / "pi_coverage_by_group.csv"
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}\n")

    cols = ["run", "RMSE", "coverage_marginal", "coverage_MDD", "coverage_HC", "PI_width"]
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nSummary across all 21 runs:")
    print(f"  Marginal mean coverage:  {out['coverage_marginal'].mean():.3f}")
    print(f"  MDD mean coverage:       {out['coverage_MDD'].mean():.3f}  "
          f"(min {out['coverage_MDD'].min():.3f}, max {out['coverage_MDD'].max():.3f})")
    print(f"  HC mean coverage:        {out['coverage_HC'].mean():.3f}  "
          f"(min {out['coverage_HC'].min():.3f}, max {out['coverage_HC'].max():.3f})")
    print(f"  Mean MDD-HC gap:         {(out['coverage_MDD'] - out['coverage_HC']).mean():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())