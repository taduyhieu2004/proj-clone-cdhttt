# Hướng dẫn từ file raw → file ML → upload dashboard

Tài liệu này giải thích **flow đầy đủ** để đi từ **file review thô (raw)** tới **file ML processed** mà dashboard Streamlit đang dùng.

---

## 1. Yêu cầu đầu vào: file raw

File raw là CSV đơn giản, mỗi dòng là 1 review, ví dụ:

```text
data/raw/sample100_raw.csv
review_id,review,rating_score,date
0,"Great food and very friendly staff.",5,2022-01-10
1,"Service was slow and my drink arrived late.",2,2022-01-15
...
```

**Bắt buộc** cần các cột:

- `review_id`: số nguyên, id của review (có thể là index).
- `review`: nội dung text của review.
- `rating_score`: điểm sao 1–5.
- `date`: ngày review (YYYY-MM-DD).

Bạn có thể dùng file mẫu mình đã tạo sẵn:

- `data/raw/sample100_raw.csv`

---

## 2. Chuẩn hoá raw → `<name>_reviews.csv` + `resumme_<name>.csv`

Script sử dụng: `scripts/prepare_dataset_from_raw.py`

Từ thư mục gốc project:

```bash
source .venv/bin/activate
cd /home/hieu/Workspace/sentiment-analysis-reviews

python3 scripts/prepare_dataset_from_raw.py \
  --name sample100 \
  --input data/raw/sample100_raw.csv
```

Tham số:

- `--name`: tên dataset (dùng làm prefix cho file sau này).
- `--input`: đường dẫn tới file raw CSV.

Kết quả sinh ra:

- `data/processed/sample100_reviews.csv`
  - đúng schema mà `src/sentiment.py` yêu cầu:
    - `review_id,review,local_guide_reviews,rating_score,service,meal_type,price_per_person_category,food_score,service_score,atmosphere_score,recommendations_list,date,avg_price_per_person`
  - nếu bạn chưa có `food_score`/`service_score`/`atmosphere_score`, script sẽ tạm gán = `rating_score`.
- `data/raw/resumme_sample100.csv`
  - phân bố sao:

```text
stars,reviews
5,xx
4,yy
3,zz
2,...
1,...
```

Hai file này là **đầu vào chuẩn** cho bước ML kế tiếp.

---

## 3. Chạy pipeline ML để tạo `<name>_ml_processed_reviews.csv`

Script sử dụng: `src/sentiment.py`

```bash
source .venv/bin/activate
cd /home/hieu/Workspace/sentiment-analysis-reviews

python3 src/sentiment.py --name sample100 --plot False
```

Trong đó `name = sample100` phải khớp với bước 2 (`sample100_reviews.csv`, `resumme_sample100.csv`).

Pipeline này sẽ:

1. **Làm sạch text** (`cleaned_review`):
   - lowercase, bỏ ký tự đặc biệt,
   - lemmatize bằng spaCy English,
   - bỏ stopwords.
2. **Gán sentiment** (`vader_sentiment`, `sentiment_label`):
   - Kết hợp `rating_score` và điểm VADER để ra `positive/neutral/negative`.
3. **Tính embedding** (`embedding`):
   - Dùng BERT đa ngôn ngữ (`bert-base-multilingual-cased`) để mã hoá mỗi review thành vector.
4. **Giảm chiều & cluster** (`pca_cluster`, `umap_cluster`):
   - PCA & UMAP → không gian 2D,
   - DBSCAN → gán cluster cho từng review.
5. **Phân tích giai đoạn điểm thấp** (`negative_periods_*`):
   - Nhóm theo thời gian (thường là tháng), tìm các kỳ điểm thấp.
6. **Trích từ/cụm từ tốt/xấu** (`common_positive_words/bigrams`, `common_negative_words/bigrams`).
7. **Sinh insights rule-based**:
   - `*_general_insights.json`: best / worst / improve (tổng quan).
   - `*_worst_periods_insights.json`: problems / improve cho các tháng “xấu”.
8. **Tạo mẫu review** cho dashboard (`*_sample_selected_reviews.csv`):
   - recent_best_reviews, recent_worst_reviews, best_reviews_sample, worst_reviews_sample, low_score_reviews.

Sau khi chạy xong, bạn sẽ có bộ file:

- `data/processed/sample100_ml_processed_reviews.csv`   ← **file chính để upload vào app**
- `data/processed/sample100_sample_selected_reviews.csv`
- `data/processed/sample100_general_insights.json`
- `data/processed/sample100_worst_periods_insights.json`
- `data/raw/resumme_sample100.csv` (đã có từ bước 2)

---

## 4. Upload lên dashboard Streamlit

Chạy app (từ thư mục gốc):

```bash
source .venv/bin/activate
cd /home/hieu/Workspace/sentiment-analysis-reviews

streamlit run app/app.py
```

Hoặc nếu đã có `run.sh`:

```bash
./run.sh
```

Trên giao diện web:

1. Ở sidebar, tìm **Select CSV File → Choose a CSV file**.
2. Chọn file:  
   **`data/processed/sample100_ml_processed_reviews.csv`**
3. App sẽ tự load kèm theo (dựa vào `name = sample100`):
   - `data/raw/resumme_sample100.csv`
   - `data/processed/sample100_sample_selected_reviews.csv`
   - `data/processed/sample100_general_insights.json`
   - `data/processed/sample100_worst_periods_insights.json`

Khi đó, toàn bộ dashboard (Status, Customer Insights, Bad times Deep Dive, ML Lab) sẽ chạy trên bộ dataset **`sample100`** của bạn.

---

## 5. Tóm tắt flow đầy đủ

1. **Chuẩn bị file raw** (ví dụ `data/raw/sample100_raw.csv` với cột `review_id, review, rating_score, date`).  
2. **Chuẩn hoá** → `sample100_reviews.csv` + `resumme_sample100.csv`:
   ```bash
   python3 scripts/prepare_dataset_from_raw.py --name sample100 --input data/raw/sample100_raw.csv
   ```
3. **Chạy ML pipeline** → sinh file ML processed & insights:
   ```bash
   python3 src/sentiment.py --name sample100 --plot False
   ```
4. **Upload lên dashboard**:
   - Chạy `streamlit run app/app.py` hoặc `./run.sh`.
   - Upload `data/processed/sample100_ml_processed_reviews.csv` trong sidebar.\n+
