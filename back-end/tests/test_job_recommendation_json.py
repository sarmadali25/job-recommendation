from datetime import datetime, timedelta

import pandas as pd

from job_recommendation_json import (
    calculate_job_freshness,
    categorize_location,
    derive_job_title_short,
    extract_salary_from_json,
    extract_skills_from_json,
    transform_json_to_dataframe,
)


def test_extract_salary_from_json_uses_average_for_yearly_salary():
    job = {"job_min_salary": 100000, "job_max_salary": 140000, "job_salary_period": "YEAR"}
    assert extract_salary_from_json(job) == 120000.0


def test_extract_salary_from_json_converts_hourly_to_yearly():
    job = {"job_min_salary": 50, "job_max_salary": 70, "job_salary_period": "HOUR"}
    # Average hourly = 60, yearly conversion assumes 40h/week * 52 weeks.
    assert extract_salary_from_json(job) == 124800.0


def test_extract_salary_from_json_returns_none_when_missing():
    assert extract_salary_from_json({"job_salary_period": "YEAR"}) is None


def test_derive_job_title_short_removes_prefix_and_suffix():
    assert derive_job_title_short("Senior Software Engineer - API") == "Software Engineer"
    assert derive_job_title_short("Staff Software Engineer, Core Infrastructure") == "Software Engineer"


def test_derive_job_title_short_handles_empty_values():
    assert derive_job_title_short(None) == "Unknown"


def test_categorize_location_maps_expected_tiers():
    assert categorize_location("San Francisco, CA") == "Tier1"
    assert categorize_location("Chicago, IL") == "Tier2"
    assert categorize_location("Remote - US") == "Remote"
    assert categorize_location("Smalltown") == "Tier3"
    assert categorize_location(pd.NA) == "Unknown"


def test_calculate_job_freshness_for_datetime_and_invalid_value():
    posted = datetime.now() - timedelta(days=7)
    assert calculate_job_freshness(posted) == 7
    assert calculate_job_freshness("not-a-date") == 365


def test_extract_skills_from_json_collects_keywords_from_highlights_and_description():
    job = {
        "job_highlights": {"Qualifications": ["Strong Python and SQL experience"]},
        "job_description": "Experience with Docker and React is a plus.",
    }
    skills = extract_skills_from_json(job)
    skills_set = set(skills)
    assert {"python", "sql", "docker", "react"}.issubset(skills_set)


def test_transform_json_to_dataframe_maps_core_fields_and_uses_job_raw_fallback():
    jobs = [
        {
            "job_id": "abc-1",
            "job_title": "Senior Data Engineer - Platform",
            "employer_name": "Acme Corp",
            "job_location": "Remote - US",
            "job_country": "US",
            "job_is_remote": True,
            "job_min_salary": 100000,
            "job_max_salary": 120000,
            "job_salary_period": "YEAR",
            "job_posted_at_datetime_utc": "2026-03-01T00:00:00.000Z",
            "job_highlights": {"Qualifications": ["Python, SQL"]},
            "job_raw": {"job_description": "Build scalable data systems with Python and SQL."},
        }
    ]

    df = transform_json_to_dataframe(jobs)
    assert df is not None
    assert len(df) == 1

    row = df.iloc[0]
    assert row["job_id"] == "abc-1"
    assert row["company_name"] == "Acme Corp"
    assert row["job_title_short"] == "Data Engineer"
    assert row["salary_year_avg"] == 110000.0
    # Pandas may return numpy.bool_ instead of built-in bool.
    assert bool(row["job_work_from_home"]) is True
    assert isinstance(row["all_skills"], list)
    assert "python" in row["all_skills"]
    # Comes from nested job_raw fallback.
    assert "scalable data systems" in row["job_description"].lower()
