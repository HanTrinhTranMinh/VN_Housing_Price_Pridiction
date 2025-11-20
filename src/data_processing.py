import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from .processing_helpers.common import load_data, drop_noisy_columns
from .processing_helpers.feature_engineering import (
    engineer_address_features,
    create_indicator_features,
    add_numeric_ratio_features,
)
from .processing_helpers.imputation import (
    impute_numerical_by_knn,
)
from .processing_helpers.encoding import one_hot_encode_categorical


def preview(df, message, n=5):
    print("\n" + "="*80)
    print("📌", message)
    print("="*80)
    print(df.head(n))
    print(f"Shape: {df.shape}\n")


def load_and_preprocess_data(csv_path):
    """
    Pipeline xử lý dữ liệu + in trạng thái sau mỗi bước.
    ĐÃ LOẠI BỎ TOÀN BỘ PHẦN NHỊ PHÂN SAI LOGIC.
    """

    df = load_data(csv_path)
    if df is None:
        return None, None, None

    print("\n===================== 📂 BẮT ĐẦU LOAD DỮ LIỆU =====================")
    print(df.info())
    print("Mô tả thống kê:\n", df.describe())
    print("Giá trị null theo cột:\n", df.isnull().sum())
    print("Các dòng đầu tiên:\n", df.head())

    # ============================================================
    # 1) TÁCH ĐẶC TRƯNG ĐỊA CHỈ
    # ============================================================
    df = engineer_address_features(df)
    preview(df, "Sau khi tách đặc trưng địa chỉ (City, District)")

    # ============================================================
    # 2) LOẠI BỎ CỘT NHIỄU
    # ============================================================
    df = drop_noisy_columns(df)
    preview(df, "Sau khi loại bỏ cột nhiễu")

    # ============================================================
    # ⚠️ LƯU Ý: BỎ HOÀN TOÀN BINARY COLUMNS (KHÔNG CÓ NHỊ PHÂN THẬT)
    # ============================================================
    binary_indicator_cols = []   # GIỮ TRỐNG - KHÔNG DÙNG

    # ============================================================
    # 3) TẠO INDICATOR CHO CỘT SỐ QUAN TRỌNG
    # ============================================================
    indicator_cols = ["Frontage", "Access Road"]
    df_indicators = create_indicator_features(df, indicator_cols)
    preview(df_indicators, "Các cột chỉ thị thiếu dữ liệu (Indicator Features)")

    # ============================================================
    # 4) PHÂN NHÓM CỘT SỐ & CỘT PHÂN LOẠI
    # ============================================================
    target_col = "Price"
    numeric_cols = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]

    categorical_candidates = ["City", "District", "Legal status", "Furniture state"]
    categorical_cols = [col for col in categorical_candidates if col in df.columns]

    df_numeric = df[numeric_cols]
    df_categorical = df[categorical_cols]

    preview(df_numeric, "Các cột số trước khi KNN impute")
    preview(df_categorical, "Các cột phân loại trước khi one-hot encode")

    # ============================================================
    # 5) XỬ LÝ TARGET y
    # ============================================================
    print("\nĐang chuyển kiểu dữ liệu cho target Price...")
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values

    # ============================================================
    # 6) ĐIỀN THIẾU NUMERIC BẰNG KNN
    # ============================================================
    df_numeric_imputed = impute_numerical_by_knn(df_numeric, numeric_cols)
    preview(df_numeric_imputed, "Sau khi điền thiếu numeric bằng KNN")

    # ============================================================
    # 7) THÊM RATIO FEATURES
    # ============================================================
    df_numeric_enhanced = add_numeric_ratio_features(df_numeric_imputed)
    preview(df_numeric_enhanced, "Sau khi thêm Ratio Features")

    numeric_feature_names = df_numeric_enhanced.columns.tolist()

    # ============================================================
    # 8) CHUẨN HÓA NUMERIC
    # ============================================================
    scaler = StandardScaler()
    df_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric_enhanced),
        columns=numeric_feature_names,
        index=df_numeric_enhanced.index
    )
    preview(df_numeric_scaled, "Sau khi chuẩn hóa StandardScaler")

    # ============================================================
    # 9) POLYNOMIAL FEATURES
    # ============================================================
    poly = PolynomialFeatures(degree=2, include_bias=False)
    numeric_poly = poly.fit_transform(df_numeric_scaled.values)

    poly_feature_names = poly.get_feature_names_out(numeric_feature_names)
    extra_feature_names = poly_feature_names[len(numeric_feature_names):]

    df_numeric_poly = pd.DataFrame(
        numeric_poly[:, len(numeric_feature_names):],
        columns=[f"poly_{name}" for name in extra_feature_names],
        index=df_numeric_scaled.index
    )
    preview(df_numeric_poly, "Đặc trưng đa thức bậc 2 (Polynomial Features)")

    # ============================================================
    # 10) ONE-HOT ENCODE CATEGORICAL
    # ============================================================
    df_categorical_safe = df_categorical.astype(str)
    df_categorical_encoded = one_hot_encode_categorical(df_categorical_safe)
    preview(df_categorical_encoded, "Sau khi One-hot encode")

    # ============================================================
    # 11) KẾT HỢP TOÀN BỘ ĐẶC TRƯNG
    # ============================================================
    X_final_df = pd.concat(
        [
            df_numeric_scaled,
            df_numeric_poly,
            df_indicators,
            df_categorical_encoded
        ],
        axis=1
    ).fillna(0)

    preview(X_final_df, "Ma trận X đầy đủ sau khi kết hợp")

    # ============================================================
    # 12) EP KIỂU FLOAT64
    # ============================================================
    if len(X_final_df.select_dtypes(include=["object"]).columns) > 0:
        print("\n⚠️ Cảnh báo: còn cột object, đang ép numeric...")
        X_final_df = X_final_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = X_final_df.to_numpy(dtype=np.float64, copy=True)
    feature_names = X_final_df.columns.tolist()

    print("\n🎯 Tổng số đặc trưng cuối cùng:", len(feature_names))
    print("🎯 Kiểu dữ liệu của X:", X.dtype)

    return X, y, feature_names
