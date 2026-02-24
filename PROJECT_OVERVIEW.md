# Project Overview – Sentiment Analysis Reviews

## What is this project?

This project is a **sentiment analysis and review dashboard** for small businesses (e.g. restaurants, shops). It helps you:

- **Collect** customer reviews (e.g. from Google Maps via scraper).
- **Clean and process** the text (NLP, sentiment, topics).
- **Visualize** results in an interactive Streamlit dashboard (trends, strengths, pain points, bad periods, ML experiments).

So you can see how customers feel over time, what they like or dislike, and where to improve.

---

## Main components

| Part | Role |
|------|------|
| **Scraper** (`src/scraper.py`) | Gets reviews from a Google Maps place URL and saves CSV (raw + star summary). |
| **Cleaning** (`src/cleaning.py`) | Parses raw scraped text: review count, star rating, dates, recommended dishes. Supports **English and Spanish** (e.g. "reviews"/"reseñas", "week"/"semana"). |
| **ML processing** (`src/ml_processing.py`) | Cleans text (English, spaCy + NLTK), sentiment (VADER + rating), embeddings (BERT), LDA topics, PCA/UMAP, DBSCAN clustering, recommendations. |
| **Insights (LLM)** (`src/llm_insights.py`) | Uses OpenAI to turn topic/sentiment data into short insights (strengths, pain points, improvements). |
| **Sentiment pipeline** (`src/sentiment.py`) | End‑to‑end: load cleaned CSV → run ML + insights → save processed files and JSON insights. |
| **Plots** (`src/plots.py`) | Plotly charts: sentiment trends, communities, k‑distance, etc. |
| **Dashboard** (`app/app.py`) | Streamlit app: upload ML CSV → see KPIs, tabs (Status, Customer Insights, Bad Times, optional ML Lab). |

---

## Data flow (short)

1. **Raw data**  
   Scraper → `data/raw/collected_reviews_<name>.csv`, `resumme_<name>.csv`.

2. **Cleaned data**  
   Notebook/cleaning → e.g. `data/processed/<name>_reviews.csv` (with parsed fields).

3. **ML + insights**  
   `python src/sentiment.py --name <place> --plot ...`  
   → `data/processed/<name>_ml_processed_reviews.csv`, `*_general_insights.json`, `*_worst_periods_insights.json`, `*_sample_selected_reviews.csv`.

4. **Dashboard**  
   Run Streamlit, upload `<name>_ml_processed_reviews.csv`; app loads the other files from `data/processed/` and `data/raw/` by name.

---

## Language: English (and scraping language)

- **NLP and sentiment** are set up for **English**:
  - spaCy: `en_core_web_sm`
  - NLTK: English stopwords, VADER (English‑oriented).
- **Cleaning** supports both **English and Spanish** for scraped text (dates, “reviews”/“stars”, “and”/“y”) so it works for Google Maps in either language.
- **LLM prompts** ask for results in **English** so insights in the dashboard are in English.

If your reviews are in English, the pipeline is aligned with that. For other languages you would need to change the spaCy model and stopwords in `src/ml_processing.py` (and optionally cleaning patterns in `src/cleaning.py`).

---

## How to run

1. **Environment**  
   Create venv, install dependencies, and download the **English** spaCy model:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt   # or install main packages manually
   python -m spacy download en_core_web_sm
   ```

2. **Dashboard**  
   From project root:
   ```bash
   streamlit run app/app.py
   ```
   Or use `./run.sh` if you have it. Then open the URL (e.g. http://localhost:8501), upload a file like `hd_ml_processed_reviews.csv` from `data/processed/`.

3. **Optional: OpenAI insights**  
   Create `openai_setup.py` with your API keys; in the app, enable “Enable OpenAI API features” to get LLM‑generated insights.

---

## File naming convention (dashboard)

The app infers the “place” from the uploaded CSV name by removing `_ml_processed_reviews` (e.g. `hd_ml_processed_reviews.csv` → place `hd`). It then looks for:

- `data/processed/<place>_general_insights.json`
- `data/processed/<place>_worst_periods_insights.json`
- `data/processed/<place>_sample_selected_reviews.csv`
- `data/raw/resumme_<place>.csv`

So keep these names in sync (e.g. all use `hd` or `Oceana Grill`) so the dashboard can load everything.

---

This file gives a high‑level explanation of the project and how it uses **English** for analysis while still supporting **English and Spanish** in scraped text cleaning.
