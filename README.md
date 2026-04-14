# Job Recommendation System

A full-stack application that pulls live job listings from the JSearch API, stores them in PostgreSQL, and recommends jobs to users using TF-IDF and cosine similarity.

## Architecture Overview

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   React UI   │──────▶│  FastAPI      │──────▶│  PostgreSQL  │
│  (Vite + TS) │◀──────│  Back-End     │◀──────│  Database    │
└──────────────┘       └──────┬───────┘       └──────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  JSearch API     │
                     │  (RapidAPI)      │
                     └──────────────────┘
```

The **React frontend** collects a user profile (skills, preferred job types, locations, salary range, remote preference) and sends it to the backend. The **FastAPI backend** then fetches job listings from the JSearch API, persists them in PostgreSQL, and runs the recommendation engine on the stored dataset. The **recommendation engine** uses scikit-learn's TF-IDF vectorizer and cosine similarity to rank jobs against the user profile and return the best matches.

## Tech Stack

| Layer            | Technologies                                                                                         |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| **Frontend**     | React 18, TypeScript, Vite 5, Tailwind CSS 3, Vitest, Testing Library                                |
| **Backend**      | Python, FastAPI, Uvicorn, Pydantic                                                                   |
| **Database**     | PostgreSQL, psycopg (v3)                                                                             |
| **ML / Data**    | pandas, NumPy, scikit-learn (TF-IDF, cosine similarity, StandardScaler, SimpleImputer, LabelEncoder) |
| **External API** | JSearch via RapidAPI                                                                                 |

## Project Structure

```
job-recommendation/
├── back-end/
│   ├── main.py                        # FastAPI application & route definitions
│   ├── db.py                          # Database connection helper (psycopg)
│   ├── setup_db.py                    # Database creation & schema migration script
│   ├── fetch_api_data.py              # JSearch API client
│   ├── jobs_repository.py             # Job CRUD operations (upsert, get all)
│   ├── job_recommendation_json.py     # Recommendation engine (TF-IDF, scoring, evaluation)
│   ├── requirements.txt               # Python dependencies
│   ├── .env                           # Environment variables (not committed)
│   └── tests/
│       └── test_job_recommendation_json.py
├── front-end/
│   ├── src/
│   │   ├── App.tsx                    # Main application component
│   │   ├── index.tsx                  # Entry point
│   │   └── utils/
│   │       └── index.ts               # Utility functions
│   ├── public/                        # Static assets
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.mjs
│   ├── tsconfig.json
│   └── postcss.config.mjs
├── .gitignore
└── README.md
```

## How It Works

### Data Ingestion

The `fetch_api_data` module calls the [JSearch API](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch) on RapidAPI to retrieve job listings. Results go through `jobs_repository.upsert_jobs()`, which maps each API response object to a row and performs an `INSERT ... ON CONFLICT DO UPDATE` (upsert) into the `jobs` table. This way, new listings get added and existing ones get updated without creating duplicates.

Each job row stores the `job_id` (primary key from JSearch), title, employer, location (city/state/country), remote status, salary range and period, O\*NET SOC code and job zone, apply options and job highlights (as JSONB), and the full raw API response (also JSONB, useful for future feature extraction).

### Recommendation Engine

The engine in `job_recommendation_json.py` follows a content-based filtering approach.

First, raw job dicts are converted into a pandas DataFrame. Fields like `employer_name` get mapped to `company_name`, skills are extracted from `job_highlights.Qualifications` and `job_description` using regex, and salaries are normalized to annual figures.

During preprocessing, missing salaries are imputed with the median for each job title category. Locations are grouped into tiers (Tier1, Tier2, Tier3, Remote). A freshness score is computed using exponential decay based on days since posting, and categorical features are label-encoded.

Then a combined text field (job title + short title + extracted skills + location) is vectorized using scikit-learn's `TfidfVectorizer` with bigrams, capped at 2,500 features.

For scoring, the user profile is projected into the same TF-IDF vector space. Cosine similarity between the user vector and all job vectors gives a text-based score. A separate feature similarity (salary, freshness, skill count, remote preference) is computed and blended at a 70/30 ratio with the text score.

Optional filters (min/max salary, location, remote-only) are applied before the final ranking.

There are also built-in evaluation functions that compute precision, recall, F1 score, and precision/recall@k against a relevance threshold, which is handy for offline testing and tuning.

### API Endpoints

| Method | Path               | Description                                                                                                                                     |
| ------ | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/`                | Health check / welcome message                                                                                                                  |
| `GET`  | `/health`          | Returns `{"status": "ok"}`                                                                                                                      |
| `GET`  | `/refresh-jobs`    | Fetches jobs from JSearch API and upserts them into the database. Accepts query params: `query`, `page`, `num_pages`, `country`, `date_posted`. |
| `GET`  | `/jobs`            | Returns all stored jobs from the database (cached in-memory for 30 minutes).                                                                    |
| `POST` | `/recommendations` | Accepts a user profile and returns ranked job recommendations.                                                                                  |

#### POST `/recommendations` Request Body

```json
{
	"user_profile": {
		"skills": ["Python", "SQL", "Docker"],
		"preferred_job_types": ["Data Engineer", "Backend Developer"],
		"preferred_locations": ["Remote", "San Francisco"],
		"salary_range": { "min": 100000, "max": 160000 },
		"remote_preference": "preferred"
	},
	"n_recommendations": 12,
	"filters": {
		"min_salary": 90000,
		"max_salary": 200000,
		"location": "San Francisco",
		"remote_only": false
	},
	"method": "content"
}
```

`n_recommendations` controls how many results to return (defaults to 12). `filters` is optional and applies hard filters before ranking. `method` can be `"content"`, `"mixed"`, or `"ranking"` (all currently resolve to content-based). `remote_preference` accepts `"no_preference"`, `"preferred"`, or `"required"`.

### Frontend

The React app is a single-page form where users fill in their skills (comma or newline separated), preferred job types (e.g. "Data Analyst", "Software Engineer"), preferred locations (e.g. "Remote", "New York"), a salary range (min/max), and their remote preference (no preference, preferred, or required).

On submission, the form POSTs to `/recommendations` and shows results in a sortable table with rank, job title, company, location, remote status, salary, and similarity score.

## Getting Started

### Prerequisites

You'll need **Python 3.11+**, **Node.js 18+** with npm (or pnpm), **PostgreSQL** (local or hosted, e.g. [Neon](https://neon.tech)), and a **RapidAPI** account with access to the [JSearch API](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch).

### Backend Setup

```bash
cd back-end

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pandas numpy scikit-learn

# Create your .env file (see Environment Variables below)
cp .env.example .env   # or create manually

# Start the server
uvicorn main:app --reload --port 8000
```

### Database Setup

The `setup_db.py` script creates the database and applies the schema automatically:

```bash
cd back-end
python setup_db.py
```

This connects to the default `postgres` database, creates the application database if it doesn't already exist, and runs `schema.sql` to set up the `jobs` table and indexes.

To seed some initial data, call the refresh endpoint once the server is running:

```bash
curl "http://localhost:8000/refresh-jobs?query=software+engineer&num_pages=3"
```

### Frontend Setup

```bash
cd front-end

# Install dependencies
npm install        # or: pnpm install

# Start the dev server
npm run dev        # Runs on http://localhost:5173 by default
```

The frontend reads `VITE_API_BASE_URL` from the environment to locate the backend. If not set, it defaults to `http://localhost:8000`.

## Environment Variables

Create a `back-end/.env` file with these variables:

| Variable        | Description                                                    | Example                                   |
| --------------- | -------------------------------------------------------------- | ----------------------------------------- |
| `DB_HOST`       | PostgreSQL host                                                | `localhost`                               |
| `DB_PORT`       | PostgreSQL port                                                | `5432`                                    |
| `DB_NAME`       | Database name                                                  | `job_recommendations`                     |
| `DB_USER`       | Database user                                                  | `postgres`                                |
| `DB_PASSWORD`   | Database password                                              | `your_password`                           |
| `DATABASE_URL`  | Full connection string (alternative to individual DB\_\* vars) | `postgresql://user:pass@host:5432/dbname` |
| `RAPID_API_KEY` | RapidAPI key for JSearch                                       | `your_rapidapi_key`                       |

For the frontend (optional):

| Variable            | Description | Default                 |
| ------------------- | ----------- | ----------------------- |
| `VITE_API_BASE_URL` | Backend URL | `http://localhost:8000` |

## API Reference

### Refresh Jobs

```
GET /refresh-jobs?query=python+developer&page=1&num_pages=1&country=all&date_posted=all
```

**Response:**

```json
{
  "message": "Jobs refreshed successfully",
  "persisted_count": 10,
  "data": [ ... ]
}
```

### List Jobs

```
GET /jobs
```

**Response:**

```json
{
  "count": 42,
  "data": [ ... ]
}
```

### Get Recommendations

```
POST /recommendations
Content-Type: application/json
```

**Response:** Each recommendation object contains:

| Field                | Type     | Description                      |
| -------------------- | -------- | -------------------------------- |
| `job_id`             | string   | Unique job identifier            |
| `rank`               | int      | Position in recommendation list  |
| `job_title`          | string   | Full job title                   |
| `job_title_short`    | string   | Normalized short title           |
| `company_name`       | string   | Employer name                    |
| `job_location`       | string   | Job location                     |
| `normalized_salary`  | float    | Annual salary estimate           |
| `job_work_from_home` | bool     | Remote availability              |
| `all_skills`         | string[] | Extracted skill keywords         |
| `similarity_score`   | float    | Content similarity score (0-1)   |
| `algorithm`          | string   | Algorithm used (`content_based`) |

## Testing

### Backend Tests

```bash
cd back-end
source .venv/bin/activate
pytest tests/ -v
```

The test suite covers salary extraction and hourly-to-yearly conversion, location tier categorization, job freshness calculation, skill extraction from highlights and descriptions, and JSON-to-DataFrame transformation with field mapping.

### Frontend Tests

```bash
cd front-end
npm test
```

```
This README.md is generated by CURSOR
```
