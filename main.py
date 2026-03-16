from fastapi import FastAPI, HTTPException
from fetch_api_data import fetch_api_data
from jobs_repository import upsert_jobs

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI backend"}


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/refresh-jobs")
def refresh_jobs(query: str = "" , page: int = 1, num_pages: int = 1, country: str = "all", date_posted: str = "all"):
    data = fetch_api_data(query, page, num_pages, country, date_posted)
    jobs = data.get("data", [])

    try:
        persisted_count = upsert_jobs(jobs)
    except Exception as exc:
        # In a real app you might log this instead of exposing details.
        raise HTTPException(status_code=500, detail=f"Failed to persist jobs: {exc}")

    return {
        "message": "Jobs refreshed successfully",
        "persisted_count": persisted_count,
        "data": jobs,
    }