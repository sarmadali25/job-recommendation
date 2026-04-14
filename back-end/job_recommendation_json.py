import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING & TRANSFORMATION FUNCTIONS
# ============================================================================

def load_data_json(json_path):
    """Load JSON file and return as list of job objects."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try to parse as JSON array first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Single object, wrap in list
                return [data]
        except json.JSONDecodeError:
            # If parsing fails, try wrapping in array brackets
            try:
                # Check if content starts with array bracket
                if content.startswith('['):
                    data = json.loads(content)
                    return data if isinstance(data, list) else [data]
                elif content.startswith('{'):
                    # Multiple objects separated by commas but missing array brackets
                    # Try wrapping in array brackets
                    data = json.loads('[' + content + ']')
                    return data
                else:
                    # Content might have leading whitespace, try wrapping
                    data = json.loads('[' + content + ']')
                    return data
            except json.JSONDecodeError as e:
                # If still fails, try parsing as JSONL (one JSON object per line)
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
    
    # Extract from job_highlights.Qualifications
    if 'job_highlights' in job_obj and job_obj['job_highlights']:
        highlights = job_obj['job_highlights']
        if 'Qualifications' in highlights and isinstance(highlights['Qualifications'], list):
            for qual in highlights['Qualifications']:
                if isinstance(qual, str):
                    # Extract technology keywords from qualification strings
                    tech_keywords = re.findall(r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Swift|Kotlin|Scala|PHP|Ruby|Perl|R|MATLAB|SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Redis|Elasticsearch|Docker|Kubernetes|AWS|Azure|GCP|React|Angular|Vue|Node\.js|Django|Flask|Spring|Express|TensorFlow|PyTorch|Scikit-learn|Pandas|NumPy|Git|Jenkins|CI/CD|REST|GraphQL|Microservices|Machine Learning|Deep Learning|Data Science|Big Data|Hadoop|Spark|Kafka|Linux|Unix|Windows|MacOS|Agile|Scrum|DevOps|Cloud|Security|Testing|QA|API|Web|Mobile|iOS|Android|Frontend|Backend|Full Stack)\b', qual, re.IGNORECASE)
                    skills.update([kw.lower() for kw in tech_keywords])
    
    # Extract from job_description
    if 'job_description' in job_obj and job_obj['job_description']:
        description = str(job_obj['job_description']).lower()
        # Common technology patterns
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
    
    # Also extract any standalone tech terms (2-4 word phrases)
    if 'job_description' in job_obj and job_obj['job_description']:
        description = str(job_obj['job_description'])
        # Extract capitalized tech terms
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
    
    # If only one is available, use that
    if min_salary is None:
        avg_salary = max_salary
    elif max_salary is None:
        avg_salary = min_salary
    else:
        avg_salary = (min_salary + max_salary) / 2
    
    # Convert to yearly if hourly
    if salary_period and salary_period.upper() == 'HOUR':
        avg_salary = avg_salary * 40 * 52  # Convert hourly to yearly
    
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
    
    # Remove common prefixes
    prefixes = ['Senior', 'Junior', 'Staff', 'Principal', 'Lead', 'Chief', 'Head', 
                'Associate', 'Entry-Level', 'Mid-Level', 'Mid', 'Sr.', 'Sr', 'Jr.', 'Jr']
    for prefix in prefixes:
        if title.startswith(prefix + ' '):
            title = title[len(prefix):].strip()
    
    # Remove suffixes after " - " or ","
    if ' - ' in title:
        title = title.split(' - ')[0].strip()
    if ',' in title:
        title = title.split(',')[0].strip()
    
    # Remove trailing parenthetical content
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
            job_effective = {**job_raw, **job}  # Prefer top-level DB fields if present.
        else:
            job_effective = job
        
        row = {}
        
        # Direct mappings
        row['job_id'] = job_effective.get('job_id', '')
        row['job_title'] = job_effective.get('job_title', '')
        row['company_name'] = job_effective.get('employer_name', '')  # Map employer_name to company_name
        row['job_location'] = job_effective.get('job_location', '')
        row['job_country'] = job_effective.get('job_country', '')
        row['job_schedule_type'] = job_effective.get('job_employment_type', '')  # Map job_employment_type
        row['job_work_from_home'] = job_effective.get('job_is_remote', False)  # Map job_is_remote
        row['job_publisher'] = job_effective.get('job_publisher', '')
        row['job_via'] = f"via {job_effective.get('job_publisher', 'Unknown')}"  # Create job_via
        row['search_location'] = job_effective.get('job_location', '')  # Use job_location as fallback
        
        # Derive job_title_short
        row['job_title_short'] = derive_job_title_short(job_effective.get('job_title', ''))
        
        # Extract skills
        all_skills = extract_skills_from_json(job_effective)
        row['job_skills'] = json.dumps(all_skills)  # Store as JSON string for compatibility
        row['job_type_skills'] = json.dumps({})  # Empty dict for compatibility
        row['all_skills'] = all_skills  # Also store as list
        
        # Extract salary
        normalized_salary = extract_salary_from_json(job_effective)
        row['salary_year_avg'] = normalized_salary
        row['salary_hour_avg'] = None
        row['salary_rate'] = job_effective.get('job_salary_period', 'YEAR')
        
        # Extract health insurance from job_benefits
        job_benefits = job_effective.get('job_benefits', [])
        if isinstance(job_benefits, list):
            row['job_health_insurance'] = 'health_insurance' in [b.lower() for b in job_benefits]
        else:
            row['job_health_insurance'] = False
        
        # Handle job_posted_date
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
                # Try parsing various date formats
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
        row['job_no_degree_mention'] = False  # Not available in JSON, set default
        
        # Store additional fields that might be useful
        row['job_description'] = job_effective.get('job_description', '')
        row['job_benefits'] = job_benefits
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def categorize_location(location):
    """Categorize location into tiers."""
    if pd.isna(location):
        return 'Unknown'
    
    location = str(location).lower()
    
    if any(city in location for city in ['san francisco', 'new york', 'seattle', 'boston', 'austin']):
        return 'Tier1'
    elif any(city in location for city in ['chicago', 'denver', 'atlanta', 'dallas', 'los angeles']):
        return 'Tier2'
    elif 'remote' in location or 'anywhere' in location:
        return 'Remote'
    else:
        return 'Tier3'

def calculate_job_freshness(posted_date):
    """Calculate days since job was posted."""
    try:
        if pd.isna(posted_date):
            return 365
        
        if isinstance(posted_date, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    job_date = datetime.strptime(posted_date, fmt)
                    break
                except:
                    continue
            else:
                return 365
        else:
            job_date = posted_date
        
        days_diff = (datetime.now() - job_date).days
        return max(0, days_diff)
    except:
        return 365

def extract_salary_info(row):
    """Extract salary information from row (for compatibility)."""
    year_avg = row.get('salary_year_avg')
    hour_avg = row.get('salary_hour_avg')
    
    if pd.notna(year_avg):
        return float(year_avg)
    elif pd.notna(hour_avg):
        return float(hour_avg) * 40 * 52
    else:
        return None

def preprocess_data_json(df):
    """
    Preprocess DataFrame from JSON data.
    Adapted from preprocess_data() to work with JSON-transformed data.
    """
    if df is None:
        return None, None, None
    
    processed_df = df.copy()
    
    # Skills are already extracted, just ensure all_skills is a list
    if 'all_skills' not in processed_df.columns:
        processed_df['all_skills'] = processed_df.apply(
            lambda x: extract_skills_from_json(x.to_dict()) if 'job_description' in x else [],
            axis=1
        )
    
    # Ensure all_skills is always a list
    processed_df['all_skills'] = processed_df['all_skills'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    processed_df['skill_count'] = processed_df['all_skills'].apply(len)
    
    # Extract salary info
    processed_df['normalized_salary'] = processed_df.apply(extract_salary_info, axis=1)
    
    # Fill missing salaries with median by job_title_short
    salary_medians = processed_df.groupby('job_title_short')['normalized_salary'].median()
    processed_df['normalized_salary'] = processed_df.apply(
        lambda x: salary_medians.get(x['job_title_short'], 75000) if pd.isna(x['normalized_salary']) else x['normalized_salary'],
        axis=1
    )
    
    # Location processing
    processed_df['location_tier'] = processed_df['job_location'].apply(categorize_location)
    processed_df['job_location'] = processed_df['job_location'].fillna('Remote')
    processed_df['search_location'] = processed_df['search_location'].fillna('Anywhere')
    
    # Date processing
    processed_df['days_since_posted'] = processed_df['job_posted_date'].apply(calculate_job_freshness)
    processed_df['freshness_score'] = np.exp(-processed_df['days_since_posted'] / 30)
    
    # Combined text for TF-IDF
    processed_df['combined_text'] = (
        processed_df['job_title'].fillna('') + ' ' +
        processed_df['job_title_short'].fillna('') + ' ' +
        processed_df['all_skills'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' +
        processed_df['job_location'].fillna('')
    ).str.lower()
    
    # Categorical encoding
    categorical_columns = ['job_title_short', 'job_schedule_type', 'job_country', 'location_tier', 'company_name']
    label_encoders = {}
    
    for col in categorical_columns:
        if col in processed_df.columns:
            le = LabelEncoder()
            processed_df[f'{col}_encoded'] = le.fit_transform(processed_df[col].astype(str))
            label_encoders[col] = le
    
    # Feature columns
    feature_columns = [
        'normalized_salary', 'skill_count', 'freshness_score',
        'job_work_from_home', 'job_health_insurance', 'days_since_posted'
    ]
    feature_columns.extend([f'{col}_encoded' for col in categorical_columns if col in processed_df.columns])
    
    feature_matrix = processed_df[feature_columns].copy()
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    feature_matrix_imputed = pd.DataFrame(
        imputer.fit_transform(feature_matrix),
        columns=feature_columns,
        index=feature_matrix.index
    )
    
    # Scale numerical columns
    numerical_cols = ['normalized_salary', 'skill_count', 'freshness_score', 'days_since_posted']
    scaler = StandardScaler()
    feature_matrix_imputed[numerical_cols] = scaler.fit_transform(feature_matrix_imputed[numerical_cols])
    
    processed_df = pd.concat([processed_df, feature_matrix_imputed.add_suffix('_scaled')], axis=1)
    
    return processed_df, scaler, label_encoders

def get_summary_stats(processed_data):
    """Get summary statistics of processed data."""
    if processed_data is None:
        return {}
    
    stats = {
        'total_jobs': len(processed_data),
        'unique_companies': processed_data['company_name'].nunique(),
        'unique_locations': processed_data['job_location'].nunique(),
        'avg_salary': processed_data['normalized_salary'].mean(),
        'salary_range': {
            'min': processed_data['normalized_salary'].min(),
            'max': processed_data['normalized_salary'].max()
        },
        'remote_jobs_percent': (processed_data['job_work_from_home'].sum() / len(processed_data)) * 100,
        'top_job_types': processed_data['job_title_short'].value_counts().head(10).to_dict(),
        'top_locations': processed_data['location_tier'].value_counts().to_dict(),
        'avg_skills_per_job': processed_data['skill_count'].mean(),
    }
    
    return stats

# ============================================================================
# CONTENT-BASED RECOMMENDATION FUNCTIONS
# ============================================================================

def create_job_content_features(job_data):
    """Create combined text features for TF-IDF vectorization."""
    job_content = (
        job_data['job_title'].fillna('').astype(str) + ' ' +
        job_data['job_title_short'].fillna('').astype(str) + ' ' +
        job_data['all_skills'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else str(x) if x else ''
        ).astype(str) + ' ' +
        job_data['company_name'].fillna('').astype(str) + ' ' +
        job_data['job_location'].fillna('').astype(str)
    ).str.lower()
    
    job_content = job_content.str.replace(r'[^\w\s+#.]', ' ', regex=True)
    job_content = job_content.str.replace(r'\s+', ' ', regex=True)
    
    return job_content

def fit_content_based_model(job_data, max_features=2500, ngram_range=(1, 2)):
    """Fit TF-IDF vectorizer and create feature matrix."""
    try:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9+#.]*\b',
            min_df=2,
            max_df=0.8
        )
        
        job_content_features = create_job_content_features(job_data)
        tfidf_matrix = tfidf_vectorizer.fit_transform(job_content_features)
        
        feature_columns = []
        feature_data = []
        
        if 'normalized_salary' in job_data.columns:
            feature_columns.append('normalized_salary')
            feature_data.append(job_data['normalized_salary'].fillna(job_data['normalized_salary'].median()))
        
        if 'freshness_score' in job_data.columns:
            feature_columns.append('freshness')
            feature_data.append(job_data['freshness_score'].fillna(0.5))
        
        if 'skill_count' in job_data.columns:
            feature_columns.append('skill_count')
            feature_data.append(job_data['skill_count'].fillna(0))
        
        if 'job_work_from_home' in job_data.columns:
            feature_columns.append('remote')
            # Source data can contain nulls for this field; coerce safely to 0/1.
            remote_series = job_data['job_work_from_home'].astype('boolean').fillna(False).astype(int)
            feature_data.append(remote_series)
        
        feature_matrix = None
        feature_scaler = StandardScaler()
        if feature_data:
            feature_matrix = np.column_stack(feature_data)
            feature_matrix = feature_scaler.fit_transform(feature_matrix)
        
        return tfidf_vectorizer, tfidf_matrix, feature_matrix, feature_scaler, feature_columns, job_data.copy()
    
    except Exception as e:
        print(f"Error in content-based recommender: {str(e)}")
        return None, None, None, None, None, None

def create_user_query_vector(user_profile, tfidf_vectorizer):
    """Create TF-IDF vector for user profile."""
    user_content_parts = []
    
    if 'skills' in user_profile:
        user_content_parts.extend(user_profile['skills'])
    
    if 'preferred_job_types' in user_profile:
        user_content_parts.extend(user_profile['preferred_job_types'])
    
    if 'preferred_locations' in user_profile:
        user_content_parts.extend(user_profile['preferred_locations'])
    
    user_query = ' '.join(user_content_parts).lower()
    user_tfidf_vector = tfidf_vectorizer.transform([user_query])
    
    return user_tfidf_vector

def create_user_feature_vector(user_profile, feature_columns, feature_scaler):
    """Create feature vector for user profile."""
    try:
        user_features = []
        
        if 'normalized_salary' in feature_columns:
            salary_range = user_profile.get('salary_range', {})
            min_salary = salary_range.get('min', 50000)
            max_salary = salary_range.get('max', 100000)
            preferred_salary = (min_salary + max_salary) / 2
            user_features.append(preferred_salary)
        
        if 'freshness' in feature_columns:
            user_features.append(1.0)
        
        if 'skill_count' in feature_columns:
            skill_count = len(user_profile.get('skills', []))
            user_features.append(skill_count)
        
        if 'remote' in feature_columns:
            remote_pref = user_profile.get('remote_preference', 'no_preference')
            user_features.append(1.0 if remote_pref in ['required', 'preferred'] else 0.0)
        
        if len(user_features) > 0:
            user_feature_vector = feature_scaler.transform([user_features])
            return user_feature_vector
        else:
            return None
    
    except Exception as e:
        print(f"Error {str(e)}")
        return None

def apply_content_filters(job_data, filters):
    """Apply filters to job data."""
    valid_indices = np.arange(len(job_data))
    
    if filters:
        if 'min_salary' in filters and filters['min_salary']:
            salary_mask = job_data['normalized_salary'] >= filters['min_salary']
            valid_indices = valid_indices[salary_mask[valid_indices]]
        
        if 'max_salary' in filters and filters['max_salary']:
            salary_mask = job_data['normalized_salary'] <= filters['max_salary']
            valid_indices = valid_indices[salary_mask[valid_indices]]
        
        if 'location' in filters and filters['location']:
            location_filter = filters['location'].lower()
            location_mask = job_data['job_location'].str.lower().str.contains(
                location_filter, na=False
            )
            valid_indices = valid_indices[location_mask[valid_indices]]
        
        if 'remote_only' in filters and filters['remote_only']:
            remote_mask = job_data['job_work_from_home'] == True
            valid_indices = valid_indices[remote_mask[valid_indices]]
    
    return valid_indices

def content_based_recommend(user_profile, tfidf_vectorizer, tfidf_matrix, feature_matrix, feature_scaler, feature_columns, job_data, n_recommendations=10, filters=None):
    """Generate content-based recommendations."""
    try:
        user_vector = create_user_query_vector(user_profile, tfidf_vectorizer)
        cosine_similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
        
        if feature_matrix is not None:
            user_features = create_user_feature_vector(user_profile, feature_columns, feature_scaler)
            if user_features is not None:
                feature_similarities = cosine_similarity(user_features, feature_matrix).flatten()
                final_similarities = 0.7 * cosine_similarities + 0.3 * feature_similarities
            else:
                final_similarities = cosine_similarities
        else:
            final_similarities = cosine_similarities
        
        valid_indices = apply_content_filters(job_data, filters)
        
        if len(valid_indices) == 0:
            return []
        
        valid_similarities = final_similarities[valid_indices]
        top_indices = valid_indices[np.argsort(valid_similarities)[::-1][:n_recommendations]]
        
        recommendations = []
        for i, job_idx in enumerate(top_indices):
            job = job_data.iloc[job_idx]
            
            recommendation = {
                'job_id': job.get('job_id', job_idx),
                'job_title': job.get('job_title', 'N/A'),
                'job_title_short': job.get('job_title_short', 'N/A'),
                'company_name': job.get('company_name', 'N/A'),
                'job_location': job.get('job_location', 'N/A'),
                'normalized_salary': job.get('normalized_salary', 0),
                'job_work_from_home': job.get('job_work_from_home', False),
                'job_health_insurance': job.get('job_health_insurance', False),
                'all_skills': job.get('all_skills', []),
                'days_since_posted': job.get('days_since_posted', 365),
                'similarity_score': float(final_similarities[job_idx]),
                'rank': i + 1,
                'algorithm': 'content_based'
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    except Exception as e:
        print(f"Error in recommendations: {str(e)}")
        return []

# ============================================================================
# MAIN RECOMMENDATION SYSTEM
# ============================================================================

def initialize_recommendation_system_json(json_path):
    """Initialize recommendation system with JSON data."""
    try:
        json_data = load_data_json(json_path)
        if json_data is None:
            raise Exception("Failed to load JSON data")
        
        df = transform_json_to_dataframe(json_data)
        if df is None:
            raise Exception("Failed to transform JSON to DataFrame")
        
        processed_data, scaler, label_encoders = preprocess_data_json(df)
        
        if processed_data is None:
            raise Exception("Error in preprocessing")
        
        tfidf_vectorizer, tfidf_matrix, feature_matrix, feature_scaler, feature_columns, content_job_data = fit_content_based_model(
            processed_data, max_features=2500, ngram_range=(1, 2)
        )
        
        stats = get_summary_stats(processed_data)
        
        return {
            'processed_data': processed_data,
            'tfidf_vectorizer': tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'feature_matrix': feature_matrix,
            'feature_scaler': feature_scaler,
            'feature_columns': feature_columns,
            'content_job_data': content_job_data,
            'stats': stats
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def initialize_recommendation_system_from_jobs(jobs):
    """
    Initialize recommendation system from an in-memory collection of job dicts.

    This is used by the API so the recommender can run on the Postgres-stored dataset.
    """
    try:
        if not jobs:
            return None

        df = transform_json_to_dataframe(jobs)
        if df is None or df.empty:
            return None

        processed_data, scaler, label_encoders = preprocess_data_json(df)
        if processed_data is None:
            return None

        tfidf_vectorizer, tfidf_matrix, feature_matrix, feature_scaler, feature_columns, _ = fit_content_based_model(
            processed_data, max_features=2500, ngram_range=(1, 2)
        )
        if tfidf_vectorizer is None or tfidf_matrix is None:
            return None

        stats = get_summary_stats(processed_data)

        # `content_job_data` is the processed_df copy returned by fit_content_based_model.
        return {
            'processed_data': processed_data,
            'tfidf_vectorizer': tfidf_vectorizer,
            'tfidf_matrix': tfidf_matrix,
            'feature_matrix': feature_matrix,
            'feature_scaler': feature_scaler,
            'feature_columns': feature_columns,
            'content_job_data': processed_data.copy(),
            'stats': stats
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def get_recommendations(system_data, user_profile, n_recommendations=12, filters=None, method='content'):
    """Get job recommendations using content-based method."""
    if system_data is None:
        return []
    
    try:
        # Ranking/mixed methods were removed; keep `method` argument for backward compatibility.
        recommendations = content_based_recommend(
            user_profile=user_profile,
            tfidf_vectorizer=system_data['tfidf_vectorizer'],
            tfidf_matrix=system_data['tfidf_matrix'],
            feature_matrix=system_data['feature_matrix'],
            feature_scaler=system_data['feature_scaler'],
            feature_columns=system_data['feature_columns'],
            job_data=system_data['content_job_data'],
            n_recommendations=n_recommendations,
            filters=filters
        )
        
        return recommendations
    
    except Exception as e:
        return []

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def calculate_job_relevance(job, user_profile, relevance_threshold=0.6):
    """Calculate relevance score for a job based on user profile."""
    relevance_score = 0.0
    user_skills = set([skill.lower().strip() for skill in user_profile.get('skills', [])])
    job_skills = set([skill.lower().strip() for skill in job.get('all_skills', [])])
    
    if user_skills:
        skill_match_ratio = len(user_skills.intersection(job_skills)) / len(user_skills)
        relevance_score += 0.4 * skill_match_ratio
    
    salary_range = user_profile.get('salary_range', {})
    job_salary = job.get('normalized_salary', 0)
    
    if salary_range and job_salary > 0:
        min_salary = salary_range.get('min', 0)
        max_salary = salary_range.get('max', float('inf'))
        
        if min_salary <= job_salary <= max_salary:
            relevance_score += 0.3
        else:
            if job_salary < min_salary:
                ratio = job_salary / min_salary if min_salary > 0 else 0
            else:
                ratio = max_salary / job_salary if job_salary > 0 else 0
            relevance_score += 0.3 * max(0, min(1, ratio))
    
    preferred_locations = [loc.lower().strip() for loc in user_profile.get('preferred_locations', [])]
    job_location = job.get('job_location', '').lower()
    remote_preference = user_profile.get('remote_preference', 'no_preference')
    
    location_match = False
    
    for pref_loc in preferred_locations:
        if pref_loc in job_location or 'remote' in pref_loc:
            location_match = True
            break
    
    if remote_preference in ['required', 'preferred'] and job.get('job_work_from_home', False):
        location_match = True
    
    if location_match:
        relevance_score += 0.2
    
    preferred_job_types = [jt.lower().strip() for jt in user_profile.get('preferred_job_types', [])]
    job_title = job.get('job_title', '').lower()
    job_title_short = job.get('job_title_short', '').lower()
    
    job_type_match = False
    for pref_type in preferred_job_types:
        if pref_type in job_title or pref_type in job_title_short:
            job_type_match = True
            break
    
    if job_type_match:
        relevance_score += 0.1
    
    return min(1.0, relevance_score)

def is_relevant(job, user_profile, relevance_threshold=0.6):
    """Check if a job is relevant to user profile."""
    return calculate_job_relevance(job, user_profile, relevance_threshold) >= relevance_threshold

def get_relevant_jobs_from_dataset(all_jobs, user_profile, relevance_threshold=0.6):
    """Get all relevant jobs from dataset."""
    relevant_jobs = []
    for idx, job in all_jobs.iterrows():
        if is_relevant(job.to_dict(), user_profile, relevance_threshold):
            relevant_jobs.append(job.to_dict())
    return relevant_jobs

def calculate_precision(recommendations, user_profile, relevance_threshold=0.6):
    """Calculate precision metric."""
    if not recommendations:
        return 0.0
    
    relevant_recommended = sum(1 for job in recommendations
                             if is_relevant(job, user_profile, relevance_threshold))
    
    precision = relevant_recommended / len(recommendations)
    return precision

def calculate_recall(recommendations, user_profile, all_jobs, relevance_threshold=0.6):
    """Calculate recall metric."""
    all_relevant_jobs = get_relevant_jobs_from_dataset(all_jobs, user_profile, relevance_threshold)
    
    if not all_relevant_jobs:
        return 0.0
    
    recommended_job_ids = set(job.get('job_id') for job in recommendations)
    
    relevant_recommended = 0
    for job in all_relevant_jobs:
        job_matches = all_jobs[
            (all_jobs['job_title'] == job.get('job_title', '')) &
            (all_jobs['company_name'] == job.get('company_name', ''))
        ]
        if not job_matches.empty and job_matches.index[0] in recommended_job_ids:
            relevant_recommended += 1
    
    recall = relevant_recommended / len(all_relevant_jobs)
    return recall

def calculate_f1_score(precision, recall):
    """Calculate F1 score."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_precision_at_k(recommendations, user_profile, relevance_threshold=0.6, k_values=[5, 10, 15]):
    """Calculate precision at k."""
    precision_at_k = {}
    for k in k_values:
        if k <= len(recommendations):
            precision_at_k[f'precision@{k}'] = calculate_precision(
                recommendations[:k], user_profile, relevance_threshold
            )
        else:
            precision_at_k[f'precision@{k}'] = calculate_precision(
                recommendations, user_profile, relevance_threshold
            )
    return precision_at_k

def calculate_recall_at_k(recommendations, user_profile, all_jobs, relevance_threshold=0.6, k_values=[5, 10, 15]):
    """Calculate recall at k."""
    recall_at_k = {}
    for k in k_values:
        if k <= len(recommendations):
            recall_at_k[f'recall@{k}'] = calculate_recall(
                recommendations[:k], user_profile, all_jobs, relevance_threshold
            )
        else:
            recall_at_k[f'recall@{k}'] = calculate_recall(
                recommendations, user_profile, all_jobs, relevance_threshold
            )
    return recall_at_k

def evaluate_recommendations(recommendations, user_profile, all_jobs, relevance_threshold=0.6):
    """Evaluate recommendation quality."""
    precision = calculate_precision(recommendations, user_profile, relevance_threshold)
    recall = calculate_recall(recommendations, user_profile, all_jobs, relevance_threshold)
    f1_score = calculate_f1_score(precision, recall)
    
    precision_at_k = calculate_precision_at_k(recommendations, user_profile, relevance_threshold)
    recall_at_k = calculate_recall_at_k(recommendations, user_profile, all_jobs, relevance_threshold)
    
    total_recommendations = len(recommendations)
    relevant_in_recommendations = sum(1 for job in recommendations
                                    if is_relevant(job, user_profile, relevance_threshold))
    total_relevant_available = len(get_relevant_jobs_from_dataset(all_jobs, user_profile, relevance_threshold))
    
    avg_relevance = np.mean([calculate_job_relevance(job, user_profile, relevance_threshold)
                           for job in recommendations]) if recommendations else 0
    
    evaluation_results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_recommendations': total_recommendations,
        'relevant_recommendations': relevant_in_recommendations,
        'total_relevant_available': total_relevant_available,
        'average_relevance_score': avg_relevance,
        **precision_at_k,
        **recall_at_k
    }
    
    return evaluation_results

def print_evaluation_report(evaluation_results):
    """Print evaluation report."""
    print("\n" + "================================")
    print("RECOMMENDATION SYSTEM EVALUATION REPORT")
    print("=======================================")
    
    print(f"Overall Performance:")
    print(f"  Precision: {evaluation_results['precision']:.3f}")
    print(f"  Recall:    {evaluation_results['recall']:.3f}")
    print(f"  F1 Score:  {evaluation_results['f1_score']:.3f}")
    
    print(f"\nDetailed Metrics:")
    print(f"  Total Recommendations: {evaluation_results['total_recommendations']}")
    print(f"  Relevant Recommendations: {evaluation_results['relevant_recommendations']}")
    print(f"  Total Relevant Available: {evaluation_results['total_relevant_available']}")
    print(f"  Average Relevance Score: {evaluation_results['average_relevance_score']:.3f}")
    
    print("======================================")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize system with JSON data
    json_file = 'recinnebder_dataset.json'
    system_data = initialize_recommendation_system_json(json_file)
    
    if system_data is None:
        print("Failed to initialize recommendation system")
    else:
        print("System initialized successfully!")
        print(f"Total jobs loaded: {system_data['stats']['total_jobs']}")
        print(f"Unique companies: {system_data['stats']['unique_companies']}")
        print(f"Average salary: ${system_data['stats']['avg_salary']:,.2f}")
        print(f"Remote jobs: {system_data['stats']['remote_jobs_percent']:.1f}%")
        
        # Example user profile
        user_profile = {
            'name': 'John Doe',
            'skills': ['Python', 'React', 'JavaScript', 'AWS', 'Docker'],
            'preferred_job_types': ['Software Engineer', 'Full Stack Developer', 'Backend Developer'],
            'preferred_locations': ['Remote', 'San Francisco, CA', 'New York, NY'],
            'salary_range': {'min': 100000, 'max': 150000},
            'remote_preference': 'preferred'
        }
        
        print("\n" + "="*50)
        print("USER PROFILE")
        print("="*50)
        print(f"Name: {user_profile['name']}")
        print(f"Skills: {', '.join(user_profile['skills'])}")
        print(f"Preferred Job Types: {', '.join(user_profile['preferred_job_types'])}")
        print(f"Preferred Locations: {', '.join(user_profile['preferred_locations'])}")
        print(f"Salary Range: ${user_profile['salary_range']['min']:,} - ${user_profile['salary_range']['max']:,}")
        print(f"Remote Preference: {user_profile['remote_preference']}")
        print("="*50)
        
        # Get recommendations
        recommendations = get_recommendations(
            system_data=system_data,
            user_profile=user_profile,
            n_recommendations=12,
            filters=None,
            method='content'
        )
        
        if recommendations:
            print(f"\n{'='*50}")
            print(f"JOB RECOMMENDATIONS FOR {user_profile['name']}")
            print(f"{'='*50}")
            print(f"{'#':<3} {'Rank':<5} {'Job Title':<35} {'Company':<25} {'Location':<20} {'Salary':<12} {'Benefits':<15}")
            print("-" * 130)
            
            for i, job in enumerate(recommendations, 1):
                salary_str = f"${job['normalized_salary']:,.0f}" if job['normalized_salary'] > 0 else "Not specified"
                
                benefits = []
                if job.get('job_work_from_home'):
                    benefits.append("Remote")
                if job.get('job_health_insurance'):
                    benefits.append("Health")
                benefits_str = ", ".join(benefits) if benefits else "None"
                
                rank = job.get('rank', i)
                title = job['job_title'][:34] + "..." if len(job['job_title']) > 34 else job['job_title']
                company = job['company_name'][:24] + "..." if len(job['company_name']) > 24 else job['company_name']
                location = job['job_location'][:19] + "..." if len(job['job_location']) > 19 else job['job_location']
                
                print(f"{i:<3} {rank:<5} {title:<35} {company:<25} {location:<20} {salary_str:<12} {benefits_str:<15}")
            
            # Detailed view of top 5
            print(f"\n{'='*50}")
            print("DETAILED VIEW - TOP 5 RECOMMENDATIONS")
            print(f"{'='*50}")
            for i, job in enumerate(recommendations[:5], 1):
                rank = job.get('rank', i)
                print(f"\n{i}. {job['job_title']} (Rank: {rank})")
                print(f"   Company: {job['company_name']}")
                print(f"   Location: {job['job_location']}")
                print(f"   Salary: ${job['normalized_salary']:,.0f}" if job['normalized_salary'] > 0 else "   Salary: Not specified")
                benefits = []
                if job.get('job_work_from_home'):
                    benefits.append("Remote Work")
                if job.get('job_health_insurance'):
                    benefits.append("Health Insurance")
                if benefits:
                    print(f"   Benefits: {', '.join(benefits)}")
                
                if job.get('all_skills') and len(job['all_skills']) > 0:
                    skills_display = ', '.join(job['all_skills'][:8])
                    if len(job['all_skills']) > 8:
                        skills_display += f" (+{len(job['all_skills']) - 8} more)"
                    print(f"   Skills: {skills_display}")
                print(f"   Similarity Score: {job['similarity_score']:.3f}")
                print(f"   Algorithm: {job['algorithm']}")
            
            # Evaluation
            evaluation_results = evaluate_recommendations(
                recommendations=recommendations,
                user_profile=user_profile,
                all_jobs=system_data['processed_data'],
                relevance_threshold=0.6
            )
            
            print_evaluation_report(evaluation_results)
        else:
            print("\nNo recommendations found.")
