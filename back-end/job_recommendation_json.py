"""Backward-compatible re-export facade.

All implementation has been split into focused modules:
  - data_loading          : JSON loading, skill/salary extraction, DataFrame transform
  - preprocessing         : Location tiers, freshness, feature engineering
  - content_recommender   : TF-IDF model, user vectors, content-based recommendations
  - evaluation            : Relevance scoring, precision, recall, F1
  - recommendation_engine : System initialisation, get_recommendations, __main__
"""

# --- data_loading ---
from data_loading import (  # noqa: F401
    extract_salary_from_json,
    extract_skills_from_json,
    load_data_json,
    derive_job_title_short,
    transform_json_to_dataframe,
)

# --- preprocessing ---
from preprocessing import (  # noqa: F401
    categorize_location,
    calculate_job_freshness,
    extract_salary_info,
    get_summary_stats,
    preprocess_data_json,
)

# --- content_recommender ---
from content_recommender import (  # noqa: F401
    apply_content_filters,
    content_based_recommend,
    create_job_content_features,
    create_user_feature_vector,
    create_user_query_vector,
    fit_content_based_model,
)

# --- evaluation ---
from evaluation import (  # noqa: F401
    calculate_f1_score,
    calculate_job_relevance,
    calculate_precision,
    calculate_precision_at_k,
    calculate_recall,
    calculate_recall_at_k,
    evaluate_recommendations,
    get_relevant_jobs_from_dataset,
    is_relevant,
    print_evaluation_report,
)

# --- recommendation_engine ---
from recommendation_engine import (  # noqa: F401
    get_recommendations,
    initialize_recommendation_system_from_jobs,
    initialize_recommendation_system_json,
)
