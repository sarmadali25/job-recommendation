-- Create the jobs table
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_title TEXT,
    employer_name TEXT,
    employer_website TEXT,
    job_location TEXT,
    job_city TEXT,
    job_state TEXT,
    job_country TEXT,
    job_is_remote BOOLEAN,
    job_posted_at_datetime_utc TIMESTAMP WITH TIME ZONE,
    job_min_salary NUMERIC,
    job_max_salary NUMERIC,
    job_salary_period TEXT,
    job_onet_soc TEXT,
    job_onet_job_zone TEXT,
    apply_options JSONB,
    job_highlights JSONB,
    job_raw JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create an index on job_posted_at_datetime_utc for faster queries
CREATE INDEX IF NOT EXISTS idx_jobs_posted_at ON jobs(job_posted_at_datetime_utc);

-- Create an index on job_country for filtering
CREATE INDEX IF NOT EXISTS idx_jobs_country ON jobs(job_country);
