from fastapi import FastAPI
from fetch_api_data import fetch_api_data

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
    return {"message": "Jobs refreshed successfully", "data": data["data"]}