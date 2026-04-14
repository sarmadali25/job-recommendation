import numpy as np


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
