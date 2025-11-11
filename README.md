# Dự đoán Giá nhà Việt Nam (Hồi quy Đa thức)

Dự án này là một mô hình Machine Learning được xây dựng từ đầu (from scratch) để dự đoán giá nhà tại Việt Nam, dựa trên bộ dữ liệu [Vietnam Housing Dataset 2024 từ Kaggle](https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024).

Mục tiêu của dự án là thực hành toàn bộ quy trình khoa học dữ liệu: từ làm sạch dữ liệu, kỹ thuật đặc trưng, đến việc tự xây dựng mô hình Hồi quy Ridge (L2) với các đặc trưng Đa thức (Polynomial Features) mà không dùng các thư viện "hộp đen" như `scikit-learn` cho chính mô hình.

## Kết quả Cuối cùng

Mô hình tốt nhất (Hồi quy Ridge với `lambda=1` và đặc trưng đa thức bậc 2) đã được tinh chỉnh và đánh giá trên tập kiểm tra (test set).

*Lưu ý: Mô hình đã được cải tiến bằng Log-Transform. Nó được huấn luyện để dự đoán `log(Price)` và `log(Area)`. Các kết quả lỗi dưới đây đã được "giải nén" (`np.expm1`) về **Tỷ VNĐ thực tế** để diễn giải.*

| Chỉ số | Kết quả (Thực tế) | Diễn giải |
| :--- | :--- | :--- |
| **R-squared ($R^2$)** | **54.43%** | Mô hình giải thích được 54.43% sự biến động của giá nhà. |
| **RMSE** | **1.4906 Tỷ VNĐ** | Lỗi trung bình, nhưng phạt nặng các dự đoán sai lệch lớn. |
| **MAE** | **1.1207 Tỷ VNĐ** | Trung bình, dự đoán của mô hình sai lệch khoảng 1.12 tỷ VNĐ. |
| **MASE** | **0.6078** | Mô hình tốt hơn 1 / 0.6078 (≈ 1.65 lần) so với mô hình "ngây thơ". |

## Các Kỹ thuật chính được áp dụng

* **Làm sạch Dữ liệu:** Xử lý `dtype('O')`, ép kiểu dữ liệu `(pd.to_numeric)`.
* **Xử lý Dữ liệu thiếu:**
    * **KNN Imputer:** Sử dụng 5 "hàng xóm" gần nhất để điền khuyết các cột số (`Frontage`, `Access Road`).
    * **Mode Imputation:** Dùng giá trị xuất hiện nhiều nhất để điền các cột chữ (`Legal status`).
* **Kỹ thuật Đặc trưng (Feature Engineering):**
    * Trích xuất `City` và `District` từ cột `Address`.
    * **One-Hot Encoding** cho tất cả các đặc trưng dạng chữ (`City`, `District`, `Legal status`...).
    * **Log-Transform (`np.log1p`):** Áp dụng cho cả `Area` và `Price` (biến `y`) để xử lý độ lệch (skewness) của dữ liệu.
    * **StandardScaler:** Chuẩn hóa tất cả các đặc trưng số.
    * **PolynomialFeatures (Bậc 2):** Tự động tạo các đặc trưng phi tuyến (ví dụ: `Area^2`, `Area * Bedrooms`) để mô hình Linear Regression có thể học các mối quan hệ phức tạp.
* **Xây dựng Mô hình:**
    * **Normal Equation (Phương trình chuẩn):** Tự xây dựng hàm tính `theta`.
    * **Hồi quy Ridge (L2 Regularization):** Thêm `lambda` (Hệ số Regularization) để chống lại Overfitting.
    * **Tinh chỉnh Siêu tham số:** Tự động chạy một vòng lặp (Grid Search) để tìm giá trị `lambda` tốt nhất.

## Cấu trúc Thư mục

```bash
HousingProject/
│
├── data/
│   └── vietnam-housing-dataset-2024.csv  (Bạn phải tự tải file này về)
│
├── src/
│   ├── __init__.py                       (Biến 'src' thành một package)
│   ├── processing_helpers/               (Thư mục chứa các hàm trợ giúp)
│   │   ├── __init__.py
│   │   ├── common.py                     (Tải dữ liệu, xóa cột)
│   │   ├── encoding.py                   (Hàm one-hot encode)
│   │   ├── feature_engineering.py        (Xử lý 'Address', tạo cột chỉ thị)
│   │   └── imputation.py                 (Hàm KNN Imputer, Mode Imputer)
│   │
│   ├── data_processing.py                (File "điều phối" pipeline xử lý)
│   ├── evaluate.py                       (Các hàm tính lỗi: R2, RMSE, MAE, MASE)
│   ├── linear_regression.py              (Hàm add_bias, normal_equation, predict)
│
├── main.py                               (File chạy chính của toàn bộ dự án)
├── requirements.txt                      (Các thư viện Python cần thiết)
├── .gitignore                            (Các file và thư mục Git sẽ bỏ qua)
