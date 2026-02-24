import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is importable no matter where this script is run from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ml_processing  # noqa: E402
from src import plots  # noqa: E402

# Load the processed and cleaned data
processed_data_path = PROJECT_ROOT / "data" / "processed"
raw_data_path = PROJECT_ROOT / "data" / "raw"

# Label mapping for interest columns and label name
label_mapping = {
    'rating_score': 'Rating',
    'food_score': 'Food',
    'service_score': 'Service',
    'atmosphere_score': 'Ambient'
}

# Parameters
number_of_words = 10
n_grams = 2
eps = 0.5
min_samples = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process sentiment analysis.')
    parser.add_argument('--name', type=str, required=True, help='Name of the dataset to process')
    parser.add_argument('--plot', type=bool, default=True, help='Whether to generate plots or not')

    args = parser.parse_args()

    name = args.name
    plot = args.plot

    reviews_pro = pd.read_csv(processed_data_path / f"{name}_reviews.csv")
    resumme_raw = pd.read_csv(raw_data_path / f"resumme_{name}.csv")

    print(resumme_raw)
    print(reviews_pro.sample(min(5, len(reviews_pro))))
    reviews = reviews_pro.copy()
    reviews.reset_index(drop=True, inplace=True)
    resumme = resumme_raw.copy()

    ## Cleaning and preprocessing
    tqdm.pandas(desc="Cleaning Reviews")
    reviews['cleaned_review'] = reviews['review'].fillna('').progress_apply(ml_processing.clean_text)

    # print(reviews[['review', 'cleaned_review']].sample(5))


    label_keys = list(label_mapping.keys())

    ## Analyze sentiment
    # Analyze sentiment with VADER
    reviews = ml_processing.analyzeSentiment(reviews)

    # Extract common positive and negative phrases
    common_positive_words = ml_processing.extractCommonWords(reviews, sentiment_label = 'positive', n = number_of_words)
    common_negative_words = ml_processing.extractCommonWords(reviews, sentiment_label = 'negative', n = number_of_words)

    print("Top Positive Words:", common_positive_words)
    print("Top Negative Words:", common_negative_words)

    # Extract common positive and negative bigrams
    common_positive_bigrams = ml_processing.extractCommonNgrams(reviews, sentiment_label='positive', n = n_grams, top_n=number_of_words)
    common_negative_bigrams = ml_processing.extractCommonNgrams(reviews, sentiment_label='negative', n = n_grams, top_n=number_of_words)

    print("Top Positive Bigrams:", common_positive_bigrams)
    print("Top Negative Bigrams:", common_negative_bigrams)

    if plot:
        plots.plotSentimentTrend(reviews, years_limit=2)

    #most_recommended, less_recommended = ml_processing.analyzeRecommendations(reviews)
    #print("Top Most Recommended:", most_recommended)
    #print("Least Recommended :", less_recommended)

    ## Calculate embeddings
    tqdm.pandas(desc="Generating Embeddings")
    reviews['embedding'] = reviews['cleaned_review'].progress_apply(ml_processing.get_embedding)

    ## Analyze embeddings
    embeddings_pca = ml_processing.calculateAndVisualizeEmbeddingsPCA(reviews, score_column = label_keys[0], plot = plot)
    embeddings_umap = ml_processing.calculateAndVisualizeEmbeddingsUMAP(reviews, plot)

    # Visualize with DBSCAN clusters
    pca_clusters = ml_processing.calculateAndVisualizeEmbeddingsPCA_with_DBSCAN(reviews, score_column = label_keys[0], eps=eps, min_samples=min_samples, plot = plot)
    umap_clusters = ml_processing.calculateAndVisualizeEmbeddingsUMAP_with_DBSCAN(reviews, eps=eps, min_samples=min_samples, plot = plot)

    # Join PCA and UMAP clusters info to reviews (explicit join on review_id to avoid merge collisions)
    if "review_id" not in reviews.columns:
        reviews = reviews.reset_index().rename(columns={"index": "review_id"})
    reviews = reviews.merge(
        pca_clusters[["review_id", "pca_cluster"]], on="review_id", how="left"
    ).merge(
        umap_clusters[["review_id", "umap_cluster"]], on="review_id", how="left"
    )

    ## Save processed reviews
    ml_processed_path = processed_data_path / f"{name}_ml_processed_reviews.csv"
    reviews.to_csv(ml_processed_path, index=False)
    print("OK! -> processed sample reviews saved at", ml_processed_path)

    ## Topics
    print('=== General topics ===')
    lda_model, topics = ml_processing.analyzeTopicsLDA(reviews)

    group_columns = ['pca_cluster', 'umap_cluster', 'sentiment_label']
    topics_dict = ml_processing.generateTopicsbyColumn(reviews, group_columns)

    # Usage
    time_period = 'month'  # Change to 'week', 'year', etc. to analyze different periods
    num_periods = 3  # Number of periods with the lowest average score to select

    # Analyze for each score type
    negative_periods_rating_reviews, low_score_periods = ml_processing.analyzeLowScores(reviews, label_keys[0], time_period, num_periods)
    negative_periods_food_reviews, _ = ml_processing.analyzeLowScores(reviews, label_keys[1], time_period, num_periods)
    negative_periods_service_reviews, _ = ml_processing.analyzeLowScores(reviews, label_keys[2], time_period, num_periods)
    negative_periods_atmosphere_reviews, _ = ml_processing.analyzeLowScores(reviews, label_keys[3], time_period, num_periods)

    negative_periods_rating_topics = ml_processing.generateTopicsPerPeriod(negative_periods_rating_reviews, label_keys[0])
    negative_periods_food_topics = ml_processing.generateTopicsPerPeriod(negative_periods_food_reviews, label_keys[1])
    negative_periods_service_topics = ml_processing.generateTopicsPerPeriod(negative_periods_service_reviews, label_keys[2])
    negative_periods_atmosphere_topics = ml_processing.generateTopicsPerPeriod(negative_periods_atmosphere_reviews, label_keys[3])

    negative_periods_topics = {**negative_periods_rating_topics, **negative_periods_food_topics, **negative_periods_service_topics, **negative_periods_atmosphere_topics}

    ## Extract outliers and painpoints
    # Join all the available information
    words_dict = {
        "common_positive_words": ml_processing.format_words(common_positive_words),
        "common_negative_words": ml_processing.format_words(common_negative_words),
        "common_positive_bigrams": ml_processing.format_words(common_positive_bigrams),
        "common_negative_bigrams": ml_processing.format_words(common_negative_bigrams)
    }
    print(words_dict)

    reviews_summary_dict = {**topics_dict, **words_dict}
    print(reviews_summary_dict)

    ## Extract reviews samples
    # Calculate total score using the three main scores
    reviews_score = reviews.copy()
    food_score_mean = np.round(reviews_score[label_keys[1]].mean(), 2) / 5
    service_score_mean = np.round(reviews_score[label_keys[2]].mean(), 2) / 5
    atmosphere_score_mean = np.round(reviews_score[label_keys[3]].mean(), 2) / 5

    reviews_score[label_keys[1]] = reviews_score[label_keys[1]].fillna(food_score_mean)
    reviews_score[label_keys[2]] = reviews_score[label_keys[2]].fillna(service_score_mean)
    reviews_score[label_keys[3]] = reviews_score[label_keys[3]].fillna(atmosphere_score_mean)

    reviews_score['total_score'] = np.round(
        reviews_score[label_keys[0]] +
        (reviews_score[label_keys[1]]/5 + reviews_score[label_keys[2]]/5 + reviews_score[label_keys[3]]/5) / 3, 2)

    # Filter not null reviews
    valid_reviews = reviews_score[reviews_score['review'].notna()]

    # Select the best and worst reviews in general
    best_reviews = valid_reviews[valid_reviews['total_score'] > 5]
    worst_reviews = valid_reviews[valid_reviews['total_score'] < 2.5]

    recent_best_reviews = best_reviews.sort_values(by='date', ascending=False)
    print('last_positive_reviews')
    print(recent_best_reviews.review)
    recent_worst_reviews = worst_reviews.sort_values(by='date', ascending=False)
    print('\nlast_negative_reviews')
    print(recent_worst_reviews.review)

    best_reviews_sample = best_reviews.sort_values(by='total_score', ascending=False)
    print('\nbest_reviews_sample')
    print(best_reviews_sample.review)
    worst_reviews_sample = worst_reviews.sort_values(by='total_score', ascending=True)
    print('\nworst_reviews_sample')
    print(worst_reviews_sample.review)

    low_score_reviews = negative_periods_rating_reviews[negative_periods_rating_reviews['review'].notna()][['month','review',label_keys[0]]]
    print('\nlow_score_reviews')
    print(low_score_reviews)
    print(low_score_periods)

    # Join all the samples
    recent_best_reviews['sample_type'] = 'recent_best_reviews'
    recent_worst_reviews['sample_type'] = 'recent_worst_reviews'
    best_reviews_sample['sample_type'] = 'best_reviews_sample'
    worst_reviews_sample['sample_type'] = 'worst_reviews_sample'
    low_score_reviews['sample_type'] = 'low_score_reviews'

    combined_reviews = pd.concat([
        recent_best_reviews,
        recent_worst_reviews,
        best_reviews_sample,
        worst_reviews_sample,
        low_score_reviews
    ])

    # Save samples
    combined_reviews.reset_index(drop=True, inplace=True)
    samples_path = processed_data_path / f"{name}_sample_selected_reviews.csv"
    combined_reviews.to_csv(samples_path, index=False)
    print("OK! -> processed sample reviews saved at", samples_path)

    # ---------------------------------------------------------------------
    # Insights generation (LLM optional)
    # ---------------------------------------------------------------------
    def local_general_insights(common_pos, common_neg):
        pos_terms = [w for w, _ in common_pos[:5]] if common_pos else []
        neg_terms = [w for w, _ in common_neg[:5]] if common_neg else []
        best = [
            f"Customers often mention: {', '.join(pos_terms[:3])}." if pos_terms else "Many customers highlight positive aspects in their reviews.",
            "Overall satisfaction is driven by strong ratings and positive sentiment.",
            "A consistent experience is suggested by recurring positive mentions."
        ]
        worst = [
            f"Common issues mentioned include: {', '.join(neg_terms[:3])}." if neg_terms else "Some customers mention issues that reduce satisfaction.",
            "Negative feedback tends to cluster around service speed and order accuracy.",
            "A few reviews indicate inconsistencies in quality."
        ]
        improve = [
            "Improve service speed and order accuracy during peak hours.",
            "Ensure consistent food temperature and presentation.",
            "Monitor recurring complaints and address root causes systematically."
        ]
        return {"best": best[:3], "worst": worst[:3], "improve": improve[:3]}

    def local_worst_periods_insights(periods):
        out = {}
        for p in periods:
            out[str(p)] = {
                "problems": [
                    "Lower customer satisfaction detected in this period.",
                    "Negative reviews indicate issues that should be investigated."
                ],
                "improve": [
                    "Review staffing and service workflow during peak times.",
                    "Audit food quality and order fulfillment consistency."
                ],
            }
        return out

    general_insights_path = processed_data_path / f"{name}_general_insights.json"
    worst_periods_insights_path = processed_data_path / f"{name}_worst_periods_insights.json"

    # LLM đã được loại bỏ – luôn dùng rule-based summaries cục bộ
    insights_general = local_general_insights(common_positive_words, common_negative_words)
    insights_worst = local_worst_periods_insights(low_score_periods)

    with open(general_insights_path, "w") as f:
        json.dump(insights_general, f, indent=2)
    print("OK! -> general insights saved at", general_insights_path)

    with open(worst_periods_insights_path, "w") as f:
        json.dump(insights_worst, f, indent=2)
    print("OK! -> worst periods insights saved at", worst_periods_insights_path)