"""Utility helpers for loading LaborIQ demo datasets."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_REPO_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DATA_DIR_CANDIDATES = [_DEFAULT_DATA_DIR, _REPO_DATA_DIR]

# The demo datasets now ship exclusively under their ``*_final.csv`` names.
# Keep the loader pointed at the updated filenames to avoid falling back to
# legacy aliases that no longer exist in the repository.
_DATASET_ALIASES = {
    "oews_final_addingcost.csv": ["oews_final_addingcost.csv", "oews_final.csv"],
    "bls_final.csv": ["bls_final.csv"],
    "glassdoor_extracted_skills.csv": ["glassdoor_extracted_skills.csv"],
    "glassdoor_final.csv": ["glassdoor_final.csv"],
}


class DataNotFoundError(FileNotFoundError):
    """Raised when a required dataset cannot be located."""


def _resolve_path(filename: str) -> Path:
    candidates = _DATASET_ALIASES.get(filename, [filename])
    for base_dir in DATA_DIR_CANDIDATES:
        for candidate in candidates:
            path = base_dir / candidate
            if path.exists():
                return path
    options = ", ".join(candidates)
    raise DataNotFoundError(
        "Dataset '{filename}' (aliases: {aliases}) is missing from: {dirs}".format(
            filename=filename,
            aliases=options,
            dirs=", ".join(str(path) for path in DATA_DIR_CANDIDATES),
        )
    )


@lru_cache(maxsize=1)
def load_oews_data() -> pd.DataFrame:
    """Load the OEWS dataset with state-level occupation insights."""
    path = _resolve_path("oews_final_addingcost.csv")
    df = pd.read_csv(path)
    rename_map = {
        "area": "location_id",
        "area_title": "state_name",
        "prim_state": "state_code",
        "i_group": "industry_group",
        "occ_code": "occupation_code",
        "occ_title": "occupation",
        "tot_emp": "total_employment",
        "jobs_1000": "jobs_per_1000",
        "h_mean": "hourly_mean",
        "h_median": "hourly_median",
        "a_mean": "average_salary",
        "a_median": "median_salary",
        "h_mean_adj": "hourly_mean_adjusted",
        "a_mean_adj": "average_salary_adjusted",
    }
    df = df.rename(columns=rename_map)
    df.columns = [col.strip().lower() for col in df.columns]
    if "state_code" in df.columns:
        df["state_code"] = df["state_code"].astype(str).str.upper()
    if "state_name" in df.columns:
        df["state_name"] = df["state_name"].astype(str).str.title()
    numeric_columns = [
        "average_salary",
        "median_salary",
        "hourly_mean",
        "hourly_median",
        "hourly_mean_adjusted",
        "total_employment",
        "jobs_per_1000",
        "average_salary_adjusted",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    csv_aliases = {
        "average_salary": "a_mean",
        "hourly_mean": "h_mean",
        "average_salary_adjusted": "a_mean_adj",
        "hourly_mean_adjusted": "h_mean_adj",
        "median_salary": "a_median",
        "total_employment": "tot_emp",
    }
    for source, alias in csv_aliases.items():
        if source in df.columns and alias not in df.columns:
            df[alias] = df[source]
    return df


@lru_cache(maxsize=1)
def load_bls_data() -> pd.DataFrame:
    """Load the BLS macro-economic dataset."""
    path = _resolve_path("bls_final.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    rename_map = {
        "HourlyEarnings": "hourly_earnings",
        "JobOpenings": "job_openings",
        "LaborForceRate": "labor_rate",
        "UnemploymentRate": "unemployment_rate",
    }
    df = df.rename(columns=rename_map)
    df.columns = [col.strip().lower() for col in df.columns]
    value_columns = [column for column in df.columns if column != "date"]
    long_form = df.melt(
        id_vars=["date"],
        value_vars=value_columns,
        var_name="series_id",
        value_name="value",
    )
    long_form["series_id"] = long_form["series_id"].str.lower()
    return long_form


def _normalise_skill_label(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return raw
    tokens = []
    for token in raw.split():
        if token.isupper():
            tokens.append(token)
        else:
            tokens.append(token.capitalize())
    return " ".join(tokens)


@lru_cache(maxsize=1)
def load_glassdoor_jobs() -> pd.DataFrame:
    """Load Glassdoor job postings with a normalized schema."""
    path = _resolve_path("glassdoor_final.csv")
    df = pd.read_csv(path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    if "research_id" not in df.columns:
        raise DataNotFoundError("Column 'research_id' missing from glassdoor_final.csv")
    df["research_id"] = pd.to_numeric(df["research_id"], errors="coerce").astype("Int64")
    return df


@lru_cache(maxsize=1)
def load_glassdoor_skill_inventory() -> pd.DataFrame:
    """Return skill facts joined with job-level metadata."""
    path = _resolve_path("glassdoor_extracted_skills.csv")
    df = pd.read_csv(path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    if "research_id" not in df.columns:
        raise DataNotFoundError("Column 'research_id' missing from glassdoor_extracted_skills.csv")
    df["research_id"] = pd.to_numeric(df["research_id"], errors="coerce").astype("Int64")
    jobs = load_glassdoor_jobs()
    merged = df.merge(jobs, on="research_id", how="left", validate="many_to_one")
    return merged


def load_glassdoor_skills() -> List[str]:
    """Load a deduplicated list of skills extracted from Glassdoor job postings."""
    inventory = load_glassdoor_skill_inventory()
    if "raw_skill" in inventory.columns:
        series = inventory["raw_skill"].dropna().astype(str)
    else:
        series = inventory.iloc[:, 0].dropna().astype(str)
    skills = {_normalise_skill_label(skill) for skill in series if skill and skill.strip()}
    skills = {skill for skill in skills if skill}
    if not skills:
        skills.update(
            {
                "Python",
                "SQL",
                "Tableau",
                "Power BI",
                "Machine Learning",
                "Communication",
            }
        )
    return sorted(skills, key=str.lower)
