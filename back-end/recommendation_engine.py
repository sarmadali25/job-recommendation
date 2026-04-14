import warnings

from content_recommender import content_based_recommend, fit_content_based_model
from data_loading import load_data_json, transform_json_to_dataframe
from evaluation import evaluate_recommendations, print_evaluation_report
from preprocessing import get_summary_stats, preprocess_data_json

warnings.filterwarnings('ignore')


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


if __name__ == "__main__":
    json_file = 'recinnebder_dataset.json'
    system_data = initialize_recommendation_system_json(json_file)

    if system_data is None:
        print("Failed to initialize recommendation system")
    else:
        print("System initialized successfully!")
        print(f"Total jobs loaded: {system_data['stats']['total_jobs']}")
        print(f"Unique companies: {system_data['stats']['unique_companies']}")
        print(f"Average salary: ${system_data['stats']['avg_salary']:,.2f}")
        print(
            f"Remote jobs: {system_data['stats']['remote_jobs_percent']:.1f}%")

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
        print(
            f"Preferred Job Types: {', '.join(user_profile['preferred_job_types'])}")
        print(
            f"Preferred Locations: {', '.join(user_profile['preferred_locations'])}")
        print(
            f"Salary Range: ${user_profile['salary_range']['min']:,} - ${user_profile['salary_range']['max']:,}")
        print(f"Remote Preference: {user_profile['remote_preference']}")
        print("="*50)

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
            print(
                f"{'#':<3} {'Rank':<5} {'Job Title':<35} {'Company':<25} {'Location':<20} {'Salary':<12} {'Benefits':<15}")
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
                title = job['job_title'][:34] + \
                    "..." if len(job['job_title']) > 34 else job['job_title']
                company = job['company_name'][:24] + \
                    "..." if len(job['company_name']
                                 ) > 24 else job['company_name']
                location = job['job_location'][:19] + \
                    "..." if len(job['job_location']
                                 ) > 19 else job['job_location']

                print(
                    f"{i:<3} {rank:<5} {title:<35} {company:<25} {location:<20} {salary_str:<12} {benefits_str:<15}")

            print(f"\n{'='*50}")
            print("DETAILED VIEW - TOP 5 RECOMMENDATIONS")
            print(f"{'='*50}")
            for i, job in enumerate(recommendations[:5], 1):
                rank = job.get('rank', i)
                print(f"\n{i}. {job['job_title']} (Rank: {rank})")
                print(f"   Company: {job['company_name']}")
                print(f"   Location: {job['job_location']}")
                print(f"   Salary: ${job['normalized_salary']:,.0f}" if job['normalized_salary']
                      > 0 else "   Salary: Not specified")
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

            evaluation_results = evaluate_recommendations(
                recommendations=recommendations,
                user_profile=user_profile,
                all_jobs=system_data['processed_data'],
                relevance_threshold=0.6
            )

            print_evaluation_report(evaluation_results)
        else:
            print("\nNo recommendations found.")
