import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fetch_api_data import fetch_api_data
from jobs_repository import get_all_jobs, upsert_jobs
from recommendation_engine import (
    get_recommendations,
    initialize_recommendation_system_from_jobs,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dev-friendly: allow requests from the Vite dev server.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_JOBS_CACHE: Optional[List[Any]] = None
_JOBS_CACHE_EXPIRES_AT_MONO: float = 0.0
_JOBS_CACHE_TTL_SEC = 30 * 60


def _invalidate_jobs_cache() -> None:
    global _JOBS_CACHE, _JOBS_CACHE_EXPIRES_AT_MONO
    _JOBS_CACHE = None
    _JOBS_CACHE_EXPIRES_AT_MONO = 0.0


def _get_cached_all_jobs() -> List[Any]:
    """Return all jobs from DB, using an in-memory cache for `_JOBS_CACHE_TTL_SEC`."""
    global _JOBS_CACHE, _JOBS_CACHE_EXPIRES_AT_MONO
    now = time.monotonic()
    if _JOBS_CACHE is not None and now < _JOBS_CACHE_EXPIRES_AT_MONO:
        return _JOBS_CACHE
    jobs = get_all_jobs()
    _JOBS_CACHE = jobs
    _JOBS_CACHE_EXPIRES_AT_MONO = now + _JOBS_CACHE_TTL_SEC
    return jobs


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI backend"}


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/refresh-jobs")
def refresh_jobs(query: str = "" , page: int = 1, num_pages: int = 1, country: str = "all", date_posted: str = "all"):
    data = fetch_api_data(query, page, num_pages, country, date_posted)
    if not isinstance(data, dict) or "data" not in data:
        
        raise HTTPException(status_code=502, detail=data)
    jobs = data["data"]

    try:
        persisted_count = upsert_jobs(jobs)
    except Exception as exc:
        # In a real app you might log this instead of exposing details.
        raise HTTPException(status_code=500, detail=f"Failed to persist jobs: {exc}")

    _invalidate_jobs_cache()

    return {
        "message": "Jobs refreshed successfully",
        "persisted_count": persisted_count,
        "data": jobs,
    }


@app.get("/jobs")
def list_jobs():
    """
    Fetch all jobs saved in the database.
    """
    try:
        jobs = _get_cached_all_jobs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch jobs: {exc}")

    return {"count": len(jobs), "data": jobs}


class UserProfile(BaseModel):
    skills: List[str] = Field(default_factory=list)
    preferred_job_types: List[str] = Field(default_factory=list)
    preferred_locations: List[str] = Field(default_factory=list)
    salary_range: Dict[str, float] = Field(default_factory=dict)  # {"min": ..., "max": ...}
    remote_preference: str = "no_preference"  # "no_preference" | "preferred" | "required"


class Filters(BaseModel):
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None
    location: Optional[str] = None
    remote_only: bool = False


class RecommendRequest(BaseModel):
    user_profile: UserProfile
    n_recommendations: int = 12
    filters: Optional[Filters] = None
    method: Literal["mixed", "content", "ranking"] = "mixed"


def _model_dump_exclude_none(obj: Any) -> Any:
    # FastAPI runs on Pydantic; support both v1/v2 `dict`/`model_dump`.
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return obj.dict(exclude_none=True)


def _convert_numpy_scalars_for_json(value: Any) -> Any:
    """
    FastAPI's JSON encoder can't serialize some numpy scalar types (e.g. `numpy.bool_`).
    Convert them to native Python types recursively.
    """
    # Local import to keep startup lightweight.
    import math
    import numpy as np

    if isinstance(value, dict):
        return {k: _convert_numpy_scalars_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_numpy_scalars_for_json(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_convert_numpy_scalars_for_json(v) for v in value)
    if isinstance(value, np.generic):
        value = value.item()

    # Starlette uses `json.dumps(..., allow_nan=False)` so `NaN`/`inf` will raise.
    # Convert them to `null` so the payload is always valid JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


@app.post("/recommendations")
def recommend_jobs(req: RecommendRequest):
    """
    Generate job recommendations from the DB-backed dataset.
    Response is a JSON array of recommendation objects.
    """
    try:
        jobs = _get_cached_all_jobs()
        system_data = initialize_recommendation_system_from_jobs(jobs)
        if system_data is None:
            return []

        user_profile = _model_dump_exclude_none(req.user_profile)
        filters = _model_dump_exclude_none(req.filters) if req.filters else None

        recommendations = get_recommendations(
            system_data=system_data,
            user_profile=user_profile,
            n_recommendations=req.n_recommendations or 12,
            filters=filters,
            method=req.method or "mixed",
        )
        return _convert_numpy_scalars_for_json(recommendations)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {exc}")

    