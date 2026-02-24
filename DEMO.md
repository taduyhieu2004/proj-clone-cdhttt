# Demo – English data

## Chạy project từ đầu

Trong thư mục gốc project:

```bash
./run.sh
```

Script sẽ:

1. Tạo virtual environment `.venv` (nếu chưa có)
2. Cài dependencies (pip + PyTorch CPU + spaCy model tiếng Anh)
3. Tạo **data demo tiếng Anh** (nếu chưa có)
4. Mở dashboard Streamlit tại **http://localhost:8501**

## Dùng data demo

1. Sau khi dashboard mở, ở **sidebar bên trái** chọn **"Choose a CSV file"**.
2. Chọn file: **`data/processed/demo_ml_processed_reviews.csv`**  
   (mở thư mục `data/processed/` trong project và chọn file này, hoặc kéo thả).
3. Dashboard sẽ tự load thêm:
   - `demo_general_insights.json`
   - `demo_worst_periods_insights.json`
   - `demo_sample_selected_reviews.csv`
   - `data/raw/resumme_demo.csv`

Toàn bộ nội dung demo là **tiếng Anh** (reviews, insights, nhãn).

## Tạo lại data demo

Nếu muốn sinh lại file demo:

```bash
source .venv/bin/activate
python scripts/generate_demo_data.py
```

Các file sau sẽ bị ghi đè:

- `data/raw/resumme_demo.csv`
- `data/processed/demo_general_insights.json`
- `data/processed/demo_worst_periods_insights.json`
- `data/processed/demo_ml_processed_reviews.csv`
- `data/processed/demo_sample_selected_reviews.csv`
