import requests
import json

url = "https://jsearch.p.rapidapi.com/search"

headers = {
    "Content-Type" : "application/json",
    "X-RapidAPI-Key": "13a09fd7a7mshcabcf204e550ef5p1a19bdjsnc588d8c33791",
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

def fetch_api_data(query: str = "" , page: int = 1, num_pages: int = 1, country: str = "all", date_posted: str = "all"):
    params = {
        "query": query,
        "page": page,
        "num_pages": num_pages,
        "country": country,
        "date_posted": date_posted,
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    # print("data: ", data)

    return data


