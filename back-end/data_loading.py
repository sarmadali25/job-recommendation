import json
import re
from datetime import datetime

import pandas as pd


def load_data_json(json_path):
    """Load JSON file and return as list of job objects."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            try:
                if content.startswith('['):
                    data = json.loads(content)
                    return data if isinstance(data, list) else [data]
                elif content.startswith('{'):
                    data = json.loads('[' + content + ']')
                    return data
                else:
                    data = json.loads('[' + content + ']')
                    return data
            except json.JSONDecodeError as e:
                jobs = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line and (line.startswith('{') or line.startswith('[')):
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, list):
                                jobs.extend(obj)
                            else:
                                jobs.append(obj)
                        except:
                            continue
                if jobs:
                    return jobs
                raise e

        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"Error in loading JSON data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def extract_skills_from_json(job_obj):
    """
    Extract skills from job_highlights.Qualifications and job_description.
    Primary source: job_highlights.Qualifications
    Secondary source: Extract tech keywords from job_description
    """
    skills = set()

    if 'job_highlights' in job_obj and job_obj['job_highlights']:
        highlights = job_obj['job_highlights']
        if 'Qualifications' in highlights and isinstance(highlights['Qualifications'], list):
            for qual in highlights['Qualifications']:
                if isinstance(qual, str):
                    tech_keywords = re.findall(r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Swift|Kotlin|Scala|PHP|Ruby|Perl|R|MATLAB|SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Redis|Elasticsearch|Docker|Kubernetes|AWS|Azure|GCP|React|Angular|Vue|Node\.js|Django|Flask|Spring|Express|TensorFlow|PyTorch|Scikit-learn|Pandas|NumPy|Git|Jenkins|CI/CD|REST|GraphQL|Microservices|Machine Learning|Deep Learning|Data Science|Big Data|Hadoop|Spark|Kafka|Linux|Unix|Windows|MacOS|Agile|Scrum|DevOps|Cloud|Security|Testing|QA|API|Web|Mobile|iOS|Android|Frontend|Backend|Full Stack)\b', qual, re.IGNORECASE)
                    skills.update([kw.lower() for kw in tech_keywords])

    if 'job_description' in job_obj and job_obj['job_description']:
        description = str(job_obj['job_description']).lower()
        tech_patterns = [
            r'\b(python|java|javascript|typescript|c\+\+|c#|go|rust|swift|kotlin|scala|php|ruby|perl|r|matlab)\b',
            r'\b(sql|nosql|mongodb|postgresql|mysql|redis|elasticsearch|dynamodb|cassandra)\b',
            r'\b(docker|kubernetes|jenkins|git|ci/cd|terraform|ansible|chef|puppet)\b',
            r'\b(aws|azure|gcp|google cloud|amazon web services|microsoft azure)\b',
            r'\b(react|angular|vue|node\.js|django|flask|spring|express|rails|laravel)\b',
            r'\b(tensorflow|pytorch|scikit-learn|pandas|numpy|keras|spark|hadoop|kafka)\b',
            r'\b(machine learning|deep learning|data science|big data|artificial intelligence|ai|ml|dl)\b',
            r'\b(rest|graphql|api|microservices|soa|web services)\b',
            r'\b(linux|unix|windows|macos|ios|android)\b',
            r'\b(agile|scrum|devops|cloud|security|testing|qa|frontend|backend|full stack)\b'
        ]

        for pattern in tech_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            skills.update([m.lower() for m in matches])

    if 'job_description' in job_obj and job_obj['job_description']:
        description = str(job_obj['job_description'])
        capitalized_tech = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', description)
        common_tech_terms = ['Python', 'Java', 'JavaScript', 'React', 'Angular', 'Vue', 'Node', 'Django',
                            'Flask', 'Spring', 'TensorFlow', 'PyTorch', 'AWS', 'Azure', 'Docker', 'Kubernetes',
                            'MongoDB', 'PostgreSQL', 'MySQL', 'Redis', 'Git', 'Jenkins', 'REST', 'GraphQL']
        for term in capitalized_tech:
            if any(tech in term for tech in common_tech_terms):
                skills.add(term.lower())

    return list(skills)


def extract_salary_from_json(job_obj):
    """
    Calculate normalized_salary from job_min_salary, job_max_salary, and job_salary_period.
    Returns average salary in yearly format.
    """
    min_salary = job_obj.get('job_min_salary')
    max_salary = job_obj.get('job_max_salary')
    salary_period = job_obj.get('job_salary_period', 'YEAR')

    if min_salary is None and max_salary is None:
        return None

    if min_salary is None:
        avg_salary = max_salary
    elif max_salary is None:
        avg_salary = min_salary
    else:
        avg_salary = (min_salary + max_salary) / 2

    if salary_period and salary_period.upper() == 'HOUR':
        avg_salary = avg_salary * 40 * 52

    return float(avg_salary) if avg_salary else None


def derive_job_title_short(job_title):
    """
    Derive short job title from full job title by removing prefixes and suffixes.
    Examples:
    - "Senior Software Engineer - API" -> "Software Engineer"
    - "Staff Software Engineer, Core Infrastructure" -> "Software Engineer"
    """
    if not job_title or pd.isna(job_title):
        return 'Unknown'

    title = str(job_title).strip()

    prefixes = ['Senior', 'Junior', 'Staff', 'Principal', 'Lead', 'Chief', 'Head',
                'Associate', 'Entry-Level', 'Mid-Level', 'Mid', 'Sr.', 'Sr', 'Jr.', 'Jr']
    for prefix in prefixes:
        if title.startswith(prefix + ' '):
            title = title[len(prefix):].strip()

    if ' - ' in title:
        title = title.split(' - ')[0].strip()
    if ',' in title:
        title = title.split(',')[0].strip()

    if '(' in title:
        title = re.sub(r'\([^)]*\)', '', title).strip()

    return title if title else 'Unknown'


def transform_json_to_dataframe(json_data):
    """
    Transform JSON job data to DataFrame with expected column structure.
    Maps JSON fields to Excel format field names.
    """
    if json_data is None:
        return None

    rows = []
    for job in json_data:
        # DB rows include `job_raw` as the full original job dict (JSONB). Some fields the
        # recommender expects (e.g. `job_description`) are not stored as top-level columns.
        job_raw = job.get('job_raw') if isinstance(job, dict) else None
        if isinstance(job_raw, dict):
            job_effective = {**job_raw, **job}
        else:
            job_effective = job

        row = {}

        row['job_id'] = job_effective.get('job_id', '')
        row['job_title'] = job_effective.get('job_title', '')
        row['company_name'] = job_effective.get('employer_name', '')
        row['job_location'] = job_effective.get('job_location', '')
        row['job_country'] = job_effective.get('job_country', '')
        row['job_schedule_type'] = job_effective.get('job_employment_type', '')
        row['job_work_from_home'] = job_effective.get('job_is_remote', False)
        row['job_publisher'] = job_effective.get('job_publisher', '')
        row['job_via'] = f"via {job_effective.get('job_publisher', 'Unknown')}"
        row['search_location'] = job_effective.get('job_location', '')

        row['job_title_short'] = derive_job_title_short(job_effective.get('job_title', ''))

        all_skills = extract_skills_from_json(job_effective)
        row['job_skills'] = json.dumps(all_skills)
        row['job_type_skills'] = json.dumps({})
        row['all_skills'] = all_skills

        normalized_salary = extract_salary_from_json(job_effective)
        row['salary_year_avg'] = normalized_salary
        row['salary_hour_avg'] = None
        row['salary_rate'] = job_effective.get('job_salary_period', 'YEAR')

        job_benefits = job_effective.get('job_benefits', [])
        if isinstance(job_benefits, list):
            row['job_health_insurance'] = 'health_insurance' in [b.lower() for b in job_benefits]
        else:
            row['job_health_insurance'] = False

        posted_date = None
        if job_effective.get('job_posted_at_datetime_utc'):
            try:
                if isinstance(job_effective['job_posted_at_datetime_utc'], str):
                    posted_date = datetime.fromisoformat(job_effective['job_posted_at_datetime_utc'].replace('Z', '+00:00'))
                else:
                    posted_date = job_effective['job_posted_at_datetime_utc']
            except:
                pass

        if posted_date is None and job_effective.get('job_posted_at_timestamp'):
            try:
                posted_date = datetime.fromtimestamp(job_effective['job_posted_at_timestamp'])
            except:
                pass

        if posted_date is None and job_effective.get('job_posted_at'):
            try:
                date_str = str(job_effective['job_posted_at'])
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        posted_date = datetime.strptime(date_str, fmt)
                        break
                    except:
                        continue
            except:
                pass

        row['job_posted_date'] = posted_date if posted_date else datetime.now()
        row['job_no_degree_mention'] = False

        row['job_description'] = job_effective.get('job_description', '')
        row['job_benefits'] = job_benefits

        rows.append(row)

    df = pd.DataFrame(rows)
    return df
