import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


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
