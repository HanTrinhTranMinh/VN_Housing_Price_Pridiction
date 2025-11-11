import pandas as pd
from sklearn.impute import KNNImputer



def coerce_binary_indicator_columns(df, binary_cols):
    """
    Ép các cột dạng nhị phân (0/1/null) về kiểu số để mô hình xử lý đúng.
    """
    present_cols = [col for col in binary_cols if col in df.columns]
    if not present_cols:
        return df

    for col in present_cols:
        print(f"Đang ép cột '{col}' về nhị phân 0/1...")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def impute_numerical_by_knn(df_numeric, numeric_cols):
    """
    Điền khuyết cột số bằng KNNImputer (kèm bước ép kiểu về số).
    """
    df_numeric_safe = df_numeric.copy()
    print("Đang ép kiểu số (coerce) cho các cột số...")
    for col in numeric_cols:
        df_numeric_safe[col] = pd.to_numeric(df_numeric_safe[col], errors="coerce")
    print("Ép kiểu số hoàn tất.")

    imputer = KNNImputer(n_neighbors=5)
    original_index = df_numeric_safe.index

    df_numeric_imputed_array = imputer.fit_transform(df_numeric_safe)
    df_numeric_imputed = pd.DataFrame(
        df_numeric_imputed_array, columns=numeric_cols, index=original_index
    )

    print("KNN Imputation hoàn tất.")
    return df_numeric_imputed
