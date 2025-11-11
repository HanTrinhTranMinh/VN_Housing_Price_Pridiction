import pandas as pd

def one_hot_encode_categorical(df_categorical):
    """Mã hóa One-Hot tất cả các cột chữ được cung cấp."""
    print("Đang mã hóa One-Hot cho các cột chữ...")
    df_categorical_encoded = pd.get_dummies(
        df_categorical, 
        drop_first=True
    )
    return df_categorical_encoded