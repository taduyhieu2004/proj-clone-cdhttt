#!/bin/bash
# Chạy project Sentiment Analysis từ đầu (setup venv, cài đặt, tạo data demo, chạy app).
# Dùng: ./run.sh   hoặc   bash run.sh

set -e
cd "$(dirname "$0")"
PROJECT_ROOT="$PWD"

echo "=== Sentiment Analysis Reviews - Run ==="

# 1. Virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Dependencies
echo "Installing dependencies (may take a few minutes)..."
pip install -q --upgrade pip
if ! pip install -q -r requirements-demo.txt 2>/dev/null; then
    pip install -q streamlit pandas numpy plotly scikit-learn matplotlib seaborn networkx nltk openai tqdm gensim umap-learn transformers huggingface-hub
fi
# PyTorch CPU (nhẹ hơn bản GPU)
python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cpu
# spaCy model tiếng Anh
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || python -m spacy download en_core_web_sm

# 3. Demo data (tiếng Anh) nếu chưa có
DEMO_ML="data/processed/demo_ml_processed_reviews.csv"
if [ ! -f "$DEMO_ML" ]; then
    echo "Generating English demo data..."
    python3 scripts/generate_demo_data.py
fi

# 4. Chạy dashboard
echo ""
echo "Dashboard: http://localhost:8501"
echo "Demo: Upload file: data/processed/demo_ml_processed_reviews.csv"
echo ""
exec streamlit run app/app.py --server.headless true
