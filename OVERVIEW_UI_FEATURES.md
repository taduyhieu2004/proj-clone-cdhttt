# Tổng quan chức năng và giao diện – Sentiment Analysis Reviews

## 1. Cấu trúc tổng thể

- **Công nghệ giao diện:** Streamlit, layout wide (full width).
- **Luồng sử dụng:** Chọn file CSV (file đã xử lý ML) → App load dữ liệu và các file đi kèm (JSON, CSV) → Hiển thị dashboard theo tên địa điểm (place) trích từ tên file.
- **Điều kiện hiển thị:** Toàn bộ nội dung chính chỉ hiện sau khi **đã upload file CSV** (file dạng `*_ml_processed_reviews.csv`). Trước đó chỉ có sidebar và dòng nhắc upload.

---

## 2. Sidebar (cột trái)

| Thành phần | Mô tả |
|------------|--------|
| **Activate ML Lab Tab** | Toggle (mặc định tắt). Bật thì xuất hiện tab thứ 4 **🧪 ML Lab** – dùng embedding, PCA, UMAP, DBSCAN (nặng, chậm hơn). |
| **Select CSV File** | Ô **Choose a CSV file** – upload file CSV đã xử lý ML (ví dụ `demo_ml_processed_reviews.csv`, `hd_ml_processed_reviews.csv`). |

Sau khi chọn file, app tự load thêm (theo tên place trích từ tên file):

- `{place}_general_insights.json`
- `{place}_worst_periods_insights.json`
- `{place}_sample_selected_reviews.csv`
- `data/raw/resumme_{place}.csv`

Nếu thiếu file nào, app hiện cảnh báo (warning) tương ứng.

---

## 3. Vùng chính (sau khi đã upload file)

### 3.1. Tiêu đề và header

- **Tiêu đề trang:** `🍴 {PLACE} 🍴` (tên viết hoa, lấy từ tên file, ví dụ DEMO, HD).
- **Điểm trung bình:** Số điểm (ví dụ 4.25) kèm chuỗi sao (⭐️) tương ứng, màu xanh, cỡ chữ lớn.
- **Bốn chỉ số (metrics) – mức độ hài lòng:**
  - **Satisfaction rate (≥4★):** % review 4–5 sao (kiểu CSAT).
  - **NPS-style score:** (5★ − 1–2★) / tổng × 100, khoảng -100 đến 100.
  - **Positive (sentiment):** Số review nhãn positive (từ VADER + rating).
  - **Negative (sentiment):** Số review nhãn negative.

### 3.2. Hai cột dưới header

- **Cột trái – 📆 Last 4 weeks:** Biểu đồ đường (Plotly) xu hướng điểm trung bình theo tuần (4 tuần gần nhất) cho Rating, Food, Service, Ambient.
- **Cột phải – ⭐ Distribution:**
  - Biểu đồ **donut:** Phân bố % theo số sao (1–5).
  - Biểu đồ **cột:** Số lượng review theo từng mức sao.

---

## 4. Các tab chính

### Tab 1: 📋 Status

| Khối | Chức năng / Giao diện |
|------|------------------------|
| **🗓️ Overview** | Biểu đồ đường: Điểm trung bình (rating) theo **năm** (vài năm gần nhất). Bên cạnh: biểu đồ cột ngang điểm trung bình từng **category** (Food, Service, Ambient). |
| **📝 Monthly Overview** | Biểu đồ xu hướng **tháng** (1 năm gần nhất) cho Rating, Food, Service, Ambient; có ghi chú các tháng điểm thấp. |
| **🤩 Recommendations** | Hai cột: **Most Recommended** (món/đề xuất được nhắc nhiều nhất, kèm số lần) và **Least Recommended** (ít được khuyên dùng). |
| **🚨 Last Reviews** | Hai bảng: **Best** (mẫu review điểm cao gần đây) và **Worst** (mẫu review điểm thấp gần đây) – cột: Date, Rating, Review, Food, Service, Ambient, Meal. |
| **📥 Export report** | Hai nút: **Download summary (CSV)** (1 dòng: place, average_rating, satisfaction_rate_pct, nps_style_score, total_reviews, positive/negative/neutral_count, date_from, date_to) và **Download reviews (CSV)** (toàn bộ bảng review đang dùng). |

---

### Tab 2: 📢 Customer Insights

| Khối | Chức năng / Giao diện |
|------|------------------------|
| **Mô tả** | Insights tổng hợp từ review: điểm mạnh, điểm yếu, hướng cải thiện (từ file JSON đã tạo sẵn). |
| **Bộ lọc** | **Start Date** và **End Date** – lọc theo khoảng thời gian. |
| **💪 Strengths / 🤬 Pain Points** | Hai cột: danh sách điểm mạnh (success) và điểm yếu (error), lấy từ `general_insights.json`. |
| **💡 Areas for Improvement** | Một cột giữa: các gợi ý cải thiện (warning), từ `general_insights.json`. |
| **💘 Sentiment** | Biểu đồ: Diễn biến **% review** theo tháng theo sentiment (positive / neutral / negative) trong khoảng thời gian đã chọn. |
| **🤓 Reviews Overview** | Hai bảng: **Best** và **Worst** – mẫu review tốt/xấu trong khoảng đã lọc (cột tương tự Last Reviews). |

---

### Tab 3: 🕵🏻‍♂️ Bad times Deep Dive

| Khối | Chức năng / Giao diện |
|------|------------------------|
| **Mô tả** | Tập trung vào các giai đoạn điểm thấp: vấn đề và hướng cải thiện theo từng tháng. |
| **Bộ lọc** | **Start Date** và **End Date** – lọc khoảng thời gian xem. |
| **📝 Overview** | Biểu đồ xu hướng điểm trung bình theo **tháng** (Rating, Food, Service, Ambient) trong khoảng đã chọn. |
| **🔍 Period details** | Danh sách theo **từng tháng** (từ `worst_periods_insights.json`), mỗi tháng là một expander chứa: **🤬 Problems** (các vấn đề) và **🔧 Areas for Improvement** (gợi ý cải thiện), kèm bảng **review mẫu** của tháng đó (low_score_reviews). |

---

### Tab 4: 🧪 ML Lab (chỉ khi bật toggle ở sidebar)

| Khối | Chức năng / Giao diện |
|------|------------------------|
| **Mô tả** | Tab nâng cao: thử nghiệm tham số ML, trực quan hóa embedding, clustering (cần file có cột **embedding**). |
| **Bộ lọc** | **Start Date** và **End Date** cho dữ liệu dùng trong tab. |
| **🫂 Sentence Communities** | Đồ thị cộng đồng câu (embedding + cosine similarity + Girvan–Newman). Nếu không có cột `embedding` thì hiện thông báo cần file từ `sentiment.py`. |
| **🧩 Dimensional reduction and clustering** | Hai biểu đồ: **PCA** (điểm theo rating) và **UMAP** (điểm theo sentiment). Nếu không có embedding thì hiện figure placeholder. |
| **k-distance** | Hai đồ thị k-distance (PCA và UMAP) để chọn tham số **K**; dùng để hỗ trợ chọn **eps** cho DBSCAN. Chỉ có ý nghĩa khi đã có embedding. |
| **DBSCAN** | Hai đồ thị clustering: **PCA + DBSCAN** và **UMAP + DBSCAN** với tham số **EPS** và **minimum samples** (number input). |
| **📚 Extract Topics** | Hai cột: **Topics theo pca_cluster** và **Topics theo umap_cluster** – danh sách topic (từ LDA) cho từng cluster. |

---

## 5. Mapping nhãn hiển thị

Trong bảng và biểu đồ, app dùng tên thân thiện:

- `rating_score` → **Rating**
- `food_score` → **Food**
- `service_score` → **Service**
- `atmosphere_score` → **Ambient**
- `meal_type` → **Meal**

---

## 6. Tóm tắt chức năng theo nhóm

| Nhóm | Chức năng |
|------|-----------|
| **Đo lường hài lòng** | Điểm trung bình, Satisfaction rate (CSAT-style), NPS-style, phân bố sao, số lượng positive/negative (sentiment). |
| **Xu hướng** | Tuần (4 tuần), tháng (1 năm), năm (vài năm); theo Rating và từng category. |
| **Insights** | Điểm mạnh / điểm yếu / cải thiện (từ JSON); giai đoạn điểm thấp theo tháng (problems + improve). |
| **Mẫu review** | Best / Worst gần đây (Status); Best / Worst theo khoảng thời gian (Customer Insights); review theo tháng điểm thấp (Bad times). |
| **Đề xuất** | Most / Least recommended (món hoặc đề xuất). |
| **Xuất dữ liệu** | Download summary CSV (chỉ số tổng hợp), Download reviews CSV (toàn bộ bảng). |
| **ML (tùy chọn)** | Sentence Communities, PCA/UMAP, k-distance, DBSCAN, Topics – dùng khi có cột embedding và bật tab ML Lab. |

---

## 7. Trạng thái khi chưa upload file

- Chỉ có sidebar (toggle ML Lab + file uploader).
- Vùng chính hiển thị dòng chữ: **"Please upload a ML processed CSV file to start."**

File nên dùng: CSV đã qua bước xử lý sentiment/ML (có ít nhất các cột như `date`, `review`, `rating_score`, `sentiment_label`, và các cột điểm food/service/atmosphere nếu có). Các file JSON và CSV bổ sung đặt đúng tên và đường dẫn theo quy ước trong README / PROJECT_OVERVIEW để app load tự động.
