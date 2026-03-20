import requests
import json
import os

url = "https://jsearch.p.rapidapi.com/search"

headers_default = {
    "Content-Type" : "application/json",
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

def fetch_api_data(query: str = "" , page: int = 1, num_pages: int = 1, country: str = "all", date_posted: str = "all"):
    rapid_api_key = os.environ.get("RAPID_API_KEY")
    if not rapid_api_key:
        raise RuntimeError("Missing RAPID_API_KEY environment variable.")

    # Build headers at call time so the env var can be changed without reloading the module.
    headers = {
        **headers_default,
        "X-RapidAPI-Key": rapid_api_key,
    }

    params = {
        "query": query,
        "page": page,
        "num_pages": num_pages,
        "country": country,
        "date_posted": date_posted,
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()


    return data


