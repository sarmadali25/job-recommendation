from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List

from psycopg import sql
from psycopg.types.json import Jsonb

from db import get_connection


JOB_COLUMNS = [
    "job_id",
    "job_title",
    "employer_name",
    "employer_website",
    "job_location",
    "job_city",
    "job_state",
    "job_country",
    "job_is_remote",
    "job_posted_at_datetime_utc",
    "job_min_salary",
    "job_max_salary",
    "job_salary_period",
    "job_onet_soc",
    "job_onet_job_zone",
    "apply_options",
    "job_highlights",
    "job_raw",
]


def _parse_datetime(value: Any) -> Any:
    if not value:
        return None
    if isinstance(value, dt.datetime):
        return value
    # Expecting an ISO8601 string like "2026-02-22T00:00:00.000Z"
    try:
        s = str(value)
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


def map_job_to_row(job: Dict[str, Any]) -> List[Any]:
    """
    Map a single job dict from the API into a list of values matching JOB_COLUMNS.
    """
    # Helper to convert dict/None to Jsonb or None
    def to_jsonb(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return Jsonb(value)
        return Jsonb(value) if value else None
    
    return [
        job.get("job_id"),
        job.get("job_title"),
        job.get("employer_name"),
        job.get("employer_website"),
        job.get("job_location"),
        job.get("job_city"),
        job.get("job_state"),
        job.get("job_country"),
        job.get("job_is_remote"),
        _parse_datetime(job.get("job_posted_at_datetime_utc")),
        job.get("job_min_salary"),
        job.get("job_max_salary"),
        job.get("job_salary_period"),
        job.get("job_onet_soc"),
        job.get("job_onet_job_zone"),
        to_jsonb(job.get("apply_options")),
        to_jsonb(job.get("job_highlights")),
        Jsonb(job),  # job_raw is the entire job dict
    ]


def upsert_jobs(jobs: Iterable[Dict[str, Any]]) -> int:
    """
    Bulk upsert a collection of job dicts into the jobs table.

    Returns the number of rows attempted (len(jobs)).
    """
    rows = [map_job_to_row(job) for job in jobs]
    if not rows:
        return 0

    placeholders = sql.SQL(", ").join(sql.Placeholder() * len(JOB_COLUMNS))
    insert_stmt = sql.SQL(
        """
        INSERT INTO jobs ({columns})
        VALUES ({placeholders})
        ON CONFLICT (job_id) DO UPDATE SET
            job_title = EXCLUDED.job_title,
            employer_name = EXCLUDED.employer_name,
            employer_website = EXCLUDED.employer_website,
            job_location = EXCLUDED.job_location,
            job_city = EXCLUDED.job_city,
            job_state = EXCLUDED.job_state,
            job_country = EXCLUDED.job_country,
            job_is_remote = EXCLUDED.job_is_remote,
            job_posted_at_datetime_utc = EXCLUDED.job_posted_at_datetime_utc,
            job_min_salary = EXCLUDED.job_min_salary,
            job_max_salary = EXCLUDED.job_max_salary,
            job_salary_period = EXCLUDED.job_salary_period,
            job_onet_soc = EXCLUDED.job_onet_soc,
            job_onet_job_zone = EXCLUDED.job_onet_job_zone,
            apply_options = EXCLUDED.apply_options,
            job_highlights = EXCLUDED.job_highlights,
            job_raw = EXCLUDED.job_raw,
            updated_at = NOW()
        """
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in JOB_COLUMNS),
        placeholders=placeholders,
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(insert_stmt, rows)
        conn.commit()

    return len(rows)

