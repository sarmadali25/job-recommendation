---
description: ML and data science functions used in the job recommendation system
globs: back-end/**/*.py
alwaysApply: false
---

# ML Functions Reference

## Content-Based Recommender (`content_recommender.py`)

- **`create_job_content_features(job_data)`** — Concatenates title, short title, skills, company, and location into a single cleaned text string per job for TF-IDF input.
- **`fit_content_based_model(job_data, max_features, ngram_range)`** — Fits a `TfidfVectorizer` (unigrams + bigrams, max 2500 features, English stop words removed) and builds the TF-IDF matrix. Also creates a supplementary numeric feature matrix (salary, freshness, skill count, remote flag) scaled with `StandardScaler`.
- **`create_user_query_vector(user_profile, tfidf_vectorizer)`** — Transforms a user's skills, preferred job types, and locations into a TF-IDF vector using the fitted vectorizer.
- **`create_user_feature_vector(user_profile, feature_columns, feature_scaler)`** — Builds and scales a numeric feature vector from user preferences (salary midpoint, freshness=1.0, skill count, remote flag).
- **`content_based_recommend(...)`** — Core recommendation function. Computes `cosine_similarity` between the user TF-IDF vector and all job vectors, blends with numeric feature similarity (70/30 weight), applies filters, and returns the top-N jobs sorted by score.

## Evaluation (`evaluation.py`)

- **`calculate_job_relevance(job, user_profile)`** — Scores a single job against a user profile on four weighted dimensions: skill overlap (40%), salary fit (30%), location match (20%), and job type match (10%). Returns a score in [0, 1].
- **`is_relevant(job, user_profile, threshold)`** — Boolean check: relevance score >= threshold (default 0.6).
- **`get_relevant_jobs_from_dataset(all_jobs, user_profile)`** — Scans the full dataset and returns all jobs meeting the relevance threshold.
- **`calculate_precision(recommendations, user_profile)`** — Precision = relevant recommended / total recommended.
- **`calculate_recall(recommendations, user_profile, all_jobs)`** — Recall = relevant recommended / total relevant in dataset.
- **`calculate_recall_at_k(recommendations, user_profile, all_jobs, k_values)`** — Recall computed at the same cut-offs.
- **`evaluate_recommendations(...)`** — Runs the full evaluation suite and returns a dict with precision, recall, F1, P@k, R@k, and average relevance score.

## Key sklearn Components Used

| Component           | Module                | Purpose                                                 |
| ------------------- | --------------------- | ------------------------------------------------------- |
| `TfidfVectorizer`   | `content_recommender` | Convert job text into sparse TF-IDF feature vectors     |
| `cosine_similarity` | `content_recommender` | Measure similarity between user and job vectors         |
| `StandardScaler`    | `content_recommender` | Normalize numeric features to zero mean / unit variance |

`This above functions are generated with help of Cursor`
