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
    coerce_binary_indicator_columns,
)
from .processing_helpers.encoding import one_hot_encode_categorical


def load_and_preprocess_data(csv_path):
    """
    Tải và xử lý dữ liệu đầu vào thành ma trận đặc trưng X (float64) và target y.
    Quy trình gồm: tách đặc trưng địa chỉ, loại bỏ cột nhiễu, điền thiếu, chuẩn hóa,
    mã hóa one-hot và kết hợp tất cả đặc trưng lại với nhau.
    """
    df = load_data(csv_path)
    if df is None:
        return None, None, None # Sửa: Trả về 3 None để khớp với main.py

    # Feature engineering từ địa chỉ (sẽ gọi phiên bản đã sửa lỗi)
    df = engineer_address_features(df)

    # Loại bỏ các cột không cần thiết
    df = drop_noisy_columns(df)

    # Các cột dạng nhị phân (0/1/null) cần ép về số
    binary_candidate_order = ["Furniture", "furniture", "Certificate", "certificate"]
    binary_indicator_cols = []
    for col in binary_candidate_order:
        if col in df.columns and col not in binary_indicator_cols:
            binary_indicator_cols.append(col)

    if binary_indicator_cols:
        df = coerce_binary_indicator_columns(df, binary_indicator_cols)

    # Cột chỉ thị thiếu dữ liệu cho một số cột quan trọng
    indicator_cols = ["Frontage", "Access Road"] + binary_indicator_cols
    df_indicators = create_indicator_features(df, indicator_cols)

    target_col = "Price"
    base_numeric_cols = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
    numeric_cols = base_numeric_cols + binary_indicator_cols
    categorical_candidates = ["City", "District", "Legal status", "Furniture state"]
    categorical_cols = []
    for col in categorical_candidates:
        if col not in df.columns:
            continue
        if is_numeric_dtype(df[col]):
            if col not in numeric_cols:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    df_numeric = df[numeric_cols]
    df_categorical = df[categorical_cols]

    print(f"Đang ép kiểu số cho cột target: {target_col}")
    # Lưu ý: Phiên bản 55% R-squared của bạn KHÔNG lấy log của y
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values

    # Điền thiếu cột số bằng KNN
    df_numeric_imputed = impute_numerical_by_knn(df_numeric, numeric_cols)
    df_numeric_enhanced = add_numeric_ratio_features(df_numeric_imputed)
    numeric_feature_names = df_numeric_enhanced.columns.tolist()

    # Chuẩn hóa cột số
    print("Đang chuẩn hóa các cột số (StandardScaler)...")
    scaler = StandardScaler()
    df_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric_enhanced),
        columns=numeric_feature_names,
        index=df_numeric_enhanced.index,
    )

    # Bổ sung đặc trưng phi tuyến bậc 2
    print("Đang tạo thêm đặc trưng phi tuyến (degree=2) cho các cột số...")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    numeric_poly = poly.fit_transform(df_numeric_scaled.values)
    poly_feature_names = poly.get_feature_names_out(numeric_feature_names)
    extra_feature_names = poly_feature_names[len(numeric_feature_names) :]
    df_numeric_poly = pd.DataFrame(
        numeric_poly[:, len(numeric_feature_names) :],
        columns=[f"poly_{name}" for name in extra_feature_names],
        index=df_numeric_scaled.index,
    )

    # Chuẩn hóa kiểu dữ liệu cột chữ và one-hot encode
    print("Đang ép kiểu 'string' cho các cột chữ...")
    df_categorical_safe = df_categorical.astype(str)

    df_categorical_encoded = one_hot_encode_categorical(df_categorical_safe)

    # Kết hợp toàn bộ đặc trưng
    print("Đang kết hợp các bộ đặc trưng...")
    X_final_df = pd.concat(
        [df_numeric_scaled, df_numeric_poly, df_indicators, df_categorical_encoded],
        axis=1,
    ).fillna(0)

    # Chốt chặn cuối cùng (rất quan trọng)
    object_cols = X_final_df.select_dtypes(include=["object"]).columns
    if len(object_cols) > 0:
        print(f"!!! CẢNH BÁO: Các cột sau vẫn là 'object': {list(object_cols)}")
        print("!!! Đang ép kiểu toàn bộ ma trận X thành số (coerce -> NaN -> 0)...")
        X_final_df = X_final_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Bảo đảm toàn bộ X ở dạng float64
    X = X_final_df.to_numpy(dtype=np.float64, copy=True)
    feature_names = X_final_df.columns.tolist()

    print(f"Kiểu dữ liệu (dtype) của ma trận X cuối cùng: {X.dtype}")
    print(f"Xử lý hoàn tất. Tổng số đặc trưng: {len(feature_names)}")

    # Trả về 3 giá trị để khớp với main.py
    return X, y, feature_names
