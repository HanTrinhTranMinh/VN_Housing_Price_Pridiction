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
from .processing_helpers.imputation import impute_numerical_by_knn
from .processing_helpers.encoding import one_hot_encode_categorical


def load_and_preprocess_data(csv_path):
    """
    Pipeline xử lý dữ liệu:
    1. Load dữ liệu
    2. Tách City/District từ Address
    3. Loại bỏ cột nhiễu
    4. Xử lý missing value và numeric
    5. Chuẩn hóa
    6. Polynomial features
    7. One-hot encoding categorical
    8. Kết hợp tất cả đặc trưng thành ma trận X
    """

    # 1. Load
    df = load_data(csv_path)
    if df is None:
        return None, None, None

    # 2. Tách City/District
    df = engineer_address_features(df)

    # 3. Loại bỏ cột không dùng
    df = drop_noisy_columns(df)

    # 4. Tạo cột chỉ thị missing cho các cột numeric quan trọng
    indicator_cols = ["Frontage", "Access Road"]
    df_indicators = create_indicator_features(df, indicator_cols)

    # 5. Xác định cột số và cột phân loại
    target_col = "Price"
    base_numeric_cols = ["Area", "Frontage", "Access Road",
                         "Floors", "Bedrooms", "Bathrooms"]

    categorical_cols = ["City", "District", "Legal status", "Furniture state"]

    # Lọc đúng những cột categorical thực sự tồn tại
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    df_numeric = df[base_numeric_cols]
    df_categorical = df[categorical_cols]

    # 6. Xử lý target
    print(f"Đang ép kiểu số cho cột target: {target_col}")
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values

    # 7. Impute missing cho numeric bằng KNN
    df_numeric_imputed = impute_numerical_by_knn(df_numeric, base_numeric_cols)

    # 8. Thêm các đặc trưng tỷ lệ (ratio features)
    df_numeric_enhanced = add_numeric_ratio_features(df_numeric_imputed)
    numeric_feature_names = df_numeric_enhanced.columns.tolist()

    # 9. Chuẩn hóa numeric
    print("Đang chuẩn hóa các cột số (StandardScaler)...")
    scaler = StandardScaler()
    df_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric_enhanced),
        columns=numeric_feature_names,
        index=df_numeric_enhanced.index
    )

    # 10. Polynomial features
    print("Đang tạo đặc trưng phi tuyến (degree=2)...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    numeric_poly = poly.fit_transform(df_numeric_scaled.values)
    poly_feature_names = poly.get_feature_names_out(numeric_feature_names)

    # Lấy phần bổ sung ngoài original features
    extra_feature_names = poly_feature_names[len(numeric_feature_names):]
    df_numeric_poly = pd.DataFrame(
        numeric_poly[:, len(numeric_feature_names):],
        columns=[f"poly_{name}" for name in extra_feature_names],
        index=df_numeric_scaled.index
    )

    # 11. One-hot encode categorical
    print("Đang mã hóa one-hot các cột categorical...")
    df_categorical_safe = df_categorical.astype(str)
    df_categorical_encoded = one_hot_encode_categorical(df_categorical_safe)

    # 12. Kết hợp toàn bộ đặc trưng
    print("Đang kết hợp các bộ đặc trưng...")
    X_final_df = pd.concat(
        [
            df_numeric_scaled,
            df_numeric_poly,
            df_indicators,
            df_categorical_encoded
        ],
        axis=1
    ).fillna(0)

    # Final cleanup
    if len(X_final_df.select_dtypes(include=['object']).columns) > 0:
        X_final_df = X_final_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    X = X_final_df.to_numpy(dtype=np.float64, copy=True)
    feature_names = X_final_df.columns.tolist()

    print(f"Kiểu dữ liệu cuối cùng của X: {X.dtype}")
    print(f"Tổng số đặc trưng: {len(feature_names)}")

    return X, y, feature_names
