#!/usr/bin/env python3
"""
Prepare project input files from a truly raw reviews CSV.

Input (raw) example: data/raw/<name>_raw_reviews.csv
Required columns: review_id, review, rating_score, date

Outputs:
  - data/processed/<name>_reviews.csv           (schema expected by src/sentiment.py)
  - data/raw/resumme_<name>.csv                (stars distribution expected by dashboard)

Run:
  python3 scripts/prepare_dataset_from_raw.py --name rawreal --input data/raw/rawreal_raw_reviews.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_resume(df_raw: pd.DataFrame) -> pd.DataFrame:
    # stars = 1..5, reviews = counts
    s = df_raw["rating_score"].astype("Int64")
    counts = s.value_counts(dropna=True).to_dict()
    rows = [{"stars": stars, "reviews": int(counts.get(stars, 0))} for stars in [5, 4, 3, 2, 1]]
    return pd.DataFrame(rows)


def build_processed_reviews(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Match the schema used by existing <name>_reviews.csv in data/processed/
    out = pd.DataFrame(
        {
            "review_id": df_raw["review_id"],
            "review": df_raw["review"],
            "local_guide_reviews": pd.NA,
            "rating_score": df_raw["rating_score"],
            "service": pd.NA,
            "meal_type": pd.NA,
            "price_per_person_category": pd.NA,
            # If you don't have category scores, a common default is to reuse rating_score.
            # You can replace these later with real extracted values.
            "food_score": df_raw["rating_score"],
            "service_score": df_raw["rating_score"],
            "atmosphere_score": df_raw["rating_score"],
            "recommendations_list": "['']",
            "date": df_raw["date"],
            "avg_price_per_person": pd.NA,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Dataset name (used for output filenames)")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw CSV (columns: review_id, review, rating_score, date)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (project_root / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Raw input not found: {input_path}")

    df_raw = pd.read_csv(input_path)
    required_cols = {"review_id", "review", "rating_score", "date"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing columns in raw input: {sorted(missing)}")

    # Basic normalization
    df_raw = df_raw.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    processed_reviews = build_processed_reviews(df_raw)
    processed_path = processed_dir / f"{args.name}_reviews.csv"
    processed_reviews.to_csv(processed_path, index=False)

    resume = build_resume(df_raw)
    resume_path = raw_dir / f"resumme_{args.name}.csv"
    resume.to_csv(resume_path, index=False)

    print("Created:")
    print(f"- {processed_path}")
    print(f"- {resume_path}")


if __name__ == "__main__":
    main()

