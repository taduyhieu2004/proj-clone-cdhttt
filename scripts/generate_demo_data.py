#!/usr/bin/env python3
"""
Generate English demo data for the Sentiment Analysis dashboard.
Creates: resumme_demo.csv, demo_general_insights.json, demo_worst_periods_insights.json,
         demo_sample_selected_reviews.csv, demo_ml_processed_reviews.csv
Run from project root: python scripts/generate_demo_data.py
"""

import os
import json
import csv
from datetime import datetime, timedelta
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# English review samples for demo (positive, neutral, negative)
REVIEWS_POSITIVE = [
    "Great food and friendly staff. Will definitely come back!",
    "Amazing experience. The burger was cooked perfectly and the atmosphere was lovely.",
    "Best brunch in town. Service was quick and the coffee was excellent.",
    "Really enjoyed our dinner. The pasta was delicious and the waiter was very helpful.",
    "Clean place, tasty food, and good value. Highly recommend.",
    "Lovely cafe with a nice vibe. The sandwiches were fresh and the staff was welcoming.",
    "Had a wonderful time. Food was hot and the service was attentive.",
    "Perfect for a quick lunch. The salad was fresh and the portion was generous.",
    "Great spot for families. Kids loved the pizza and we loved the calm atmosphere.",
    "Excellent service and tasty dishes. Will be back soon.",
]

REVIEWS_NEUTRAL = [
    "Food was okay. Nothing special but not bad either.",
    "Decent place. Service could be a bit faster.",
    "Average experience. The menu is limited but prices are fair.",
    "It was fine. Maybe we ordered the wrong thing.",
    "Standard cafe. Good for a quick bite.",
]

REVIEWS_NEGATIVE = [
    "Disappointed. The food was cold and the wait was too long.",
    "Service was slow and the order came wrong. Had to send it back.",
    "Not worth the price. Portions were small and the place was noisy.",
    "Would not recommend. The burger was overcooked and the fries were soggy.",
    "Poor experience. Staff seemed rushed and forgot our drinks.",
]

MEAL_TYPES = ["Breakfast", "Brunch", "Lunch", "Dinner", "Take Away", ""]


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def write_resumme_demo():
    path = os.path.join(RAW_DIR, "resumme_demo.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stars", "reviews"])
        w.writerows([[5, 95], [4, 72], [3, 28], [2, 12], [1, 8]])
    print(f"  {path}")


def write_general_insights():
    path = os.path.join(PROCESSED_DIR, "demo_general_insights.json")
    data = {
        "best": [
            "Customers consistently praise the food quality and friendly staff.",
            "The atmosphere and value for money are frequently highlighted as strengths.",
            "Brunch and breakfast options are well received.",
        ],
        "worst": [
            "Some customers report slow service during peak hours.",
            "Food temperature and portion size have been mentioned as issues.",
            "Noise level can be high at busy times.",
        ],
        "improve": [
            "Improve service speed and order accuracy during rush hours.",
            "Ensure food is served at the right temperature and portions are consistent.",
            "Consider sound dampening to improve comfort.",
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  {path}")


def write_worst_periods_insights():
    path = os.path.join(PROCESSED_DIR, "demo_worst_periods_insights.json")
    data = {
        "2024-01": {
            "problems": [
                "Service was often slow and orders were delayed.",
                "Several customers complained about cold food.",
            ],
            "improve": [
                "Add more staff during lunch rush and improve kitchen timing.",
                "Review food holding and plating procedures.",
            ],
        },
        "2024-03": {
            "problems": [
                "Noise level was mentioned as an issue.",
                "Some dissatisfaction with portion sizes.",
            ],
            "improve": [
                "Consider layout or acoustic adjustments for busy periods.",
                "Clarify portion sizes on the menu or with staff.",
            ],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  {path}")


def random_date(start_year=2023, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")


def build_ml_rows(n=120):
    rows = []
    for i in range(n):
        r = random.random()
        if r < 0.5:
            review = random.choice(REVIEWS_POSITIVE)
            rating = random.choice([4, 5])
            sentiment = "positive"
        elif r < 0.75:
            review = random.choice(REVIEWS_NEUTRAL)
            rating = 3
            sentiment = "neutral"
        else:
            review = random.choice(REVIEWS_NEGATIVE)
            rating = random.choice([1, 2])
            sentiment = "negative"
        food = rating if random.random() > 0.2 else max(1, rating - 1)
        service = rating if random.random() > 0.25 else max(1, rating - 1)
        atmosphere = rating if random.random() > 0.3 else max(1, rating - 1)
        date = random_date()
        month = date[:7]
        cleaned = " ".join(review.lower().replace(".", "").replace(",", "").split()[:12])
        vader = 0.2 if sentiment == "positive" else (-0.2 if sentiment == "negative" else 0.0)
        rows.append({
            "review_id": i,
            "review": review,
            "local_guide_reviews": "" if random.random() > 0.3 else random.randint(1, 50),
            "rating_score": rating,
            "service": "",
            "meal_type": random.choice(MEAL_TYPES),
            "price_per_person_category": "",
            "food_score": food,
            "service_score": service,
            "atmosphere_score": atmosphere,
            "recommendations_list": "['']",
            "date": date,
            "avg_price_per_person": "",
            "cleaned_review": cleaned,
            "vader_sentiment": vader,
            "sentiment_label": sentiment,
            "pca_cluster": "" if random.random() > 0.7 else random.randint(0, 2),
            "umap_cluster": "" if random.random() > 0.7 else random.randint(0, 2),
            "month": month,
            "year": date[:4],
            "total_score": (food + service + atmosphere) / 3,
        })
    return rows


def write_ml_processed_reviews():
    path = os.path.join(PROCESSED_DIR, "demo_ml_processed_reviews.csv")
    rows = build_ml_rows(120)
    cols = [
        "review_id", "review", "local_guide_reviews", "rating_score", "service", "meal_type",
        "price_per_person_category", "food_score", "service_score", "atmosphere_score",
        "recommendations_list", "date", "avg_price_per_person", "cleaned_review",
        "vader_sentiment", "sentiment_label", "pca_cluster", "umap_cluster",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"  {path}")
    return rows


def write_sample_selected_reviews(ml_rows):
    path = os.path.join(PROCESSED_DIR, "demo_sample_selected_reviews.csv")
    by_month = {}
    for r in ml_rows:
        m = r["month"]
        if m not in by_month:
            by_month[m] = []
        by_month[m].append(r)
    sample_rows = []
    # recent_best_reviews: high rating, recent
    recent = [r for r in ml_rows if r["rating_score"] >= 4][-10:]
    for r in recent:
        row = {**r, "sample_type": "recent_best_reviews"}
        sample_rows.append(row)
    # recent_worst_reviews
    worst = [r for r in ml_rows if r["rating_score"] <= 2][-10:]
    for r in worst:
        row = {**r, "sample_type": "recent_worst_reviews"}
        sample_rows.append(row)
    # best_reviews_sample
    best = [r for r in ml_rows if r["rating_score"] >= 4][:15]
    for r in best:
        sample_rows.append({**r, "sample_type": "best_reviews_sample"})
    # worst_reviews_sample
    worst_s = [r for r in ml_rows if r["rating_score"] <= 2][:15]
    for r in worst_s:
        sample_rows.append({**r, "sample_type": "worst_reviews_sample"})
    # low_score_reviews per month (for worst periods)
    for month in ["2024-01", "2024-03"]:
        in_month = [r for r in ml_rows if r["month"] == month and r["rating_score"] <= 3]
        for r in in_month[:5]:
            sample_rows.append({**r, "sample_type": "low_score_reviews"})
    cols = [
        "review_id", "review", "local_guide_reviews", "rating_score", "service", "meal_type",
        "price_per_person_category", "food_score", "service_score", "atmosphere_score",
        "recommendations_list", "date", "avg_price_per_person", "cleaned_review",
        "vader_sentiment", "sentiment_label", "pca_cluster", "umap_cluster",
        "month", "year", "total_score", "sample_type",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sample_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"  {path}")


def main():
    print("Generating English demo data...")
    ensure_dirs()
    write_resumme_demo()
    write_general_insights()
    write_worst_periods_insights()
    ml_rows = write_ml_processed_reviews()
    write_sample_selected_reviews(ml_rows)
    print("Done. Use file: data/processed/demo_ml_processed_reviews.csv in the dashboard.")


if __name__ == "__main__":
    main()
