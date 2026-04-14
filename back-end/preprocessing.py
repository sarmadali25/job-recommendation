from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from data_loading import extract_skills_from_json


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

    if 'all_skills' not in processed_df.columns:
        processed_df['all_skills'] = processed_df.apply(
            lambda x: extract_skills_from_json(x.to_dict()) if 'job_description' in x else [],
            axis=1
        )

    processed_df['all_skills'] = processed_df['all_skills'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    processed_df['skill_count'] = processed_df['all_skills'].apply(len)

    processed_df['normalized_salary'] = processed_df.apply(extract_salary_info, axis=1)

    salary_medians = processed_df.groupby('job_title_short')['normalized_salary'].median()
    processed_df['normalized_salary'] = processed_df.apply(
        lambda x: salary_medians.get(x['job_title_short'], 75000) if pd.isna(x['normalized_salary']) else x['normalized_salary'],
        axis=1
    )

    processed_df['location_tier'] = processed_df['job_location'].apply(categorize_location)
    processed_df['job_location'] = processed_df['job_location'].fillna('Remote')
    processed_df['search_location'] = processed_df['search_location'].fillna('Anywhere')

    processed_df['days_since_posted'] = processed_df['job_posted_date'].apply(calculate_job_freshness)
    processed_df['freshness_score'] = np.exp(-processed_df['days_since_posted'] / 30)

    processed_df['combined_text'] = (
        processed_df['job_title'].fillna('') + ' ' +
        processed_df['job_title_short'].fillna('') + ' ' +
        processed_df['all_skills'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' +
        processed_df['job_location'].fillna('')
    ).str.lower()

    categorical_columns = ['job_title_short', 'job_schedule_type', 'job_country', 'location_tier', 'company_name']
    label_encoders = {}

    for col in categorical_columns:
        if col in processed_df.columns:
            le = LabelEncoder()
            processed_df[f'{col}_encoded'] = le.fit_transform(processed_df[col].astype(str))
            label_encoders[col] = le

    feature_columns = [
        'normalized_salary', 'skill_count', 'freshness_score',
        'job_work_from_home', 'job_health_insurance', 'days_since_posted'
    ]
    feature_columns.extend([f'{col}_encoded' for col in categorical_columns if col in processed_df.columns])

    feature_matrix = processed_df[feature_columns].copy()

    imputer = SimpleImputer(strategy='median')
    feature_matrix_imputed = pd.DataFrame(
        imputer.fit_transform(feature_matrix),
        columns=feature_columns,
        index=feature_matrix.index
    )

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
