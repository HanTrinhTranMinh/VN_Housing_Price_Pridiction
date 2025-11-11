import numpy as np
import pandas as pd

def engineer_address_features(df):
    """Tách 'City' và 'District' từ 'Address'."""
    print("Đang xử lý cột 'Address'...")
    address_parts = df['Address'].fillna('').str.split(', ')
    df['City'] = address_parts.str[-1].fillna('Unknown')
    df['District'] = address_parts.str[-2].fillna('Unknown')
    print("Xử lý 'Address' thành 'City' và 'District' thành công.")
    return df

def create_indicator_features(df, cols_for_indicator):
    """Tạo các cột '_Is_Missing'."""
    print("Tạo cột chỉ thị 'Is_Missing'...")
    df_indicators = pd.DataFrame()
    for col in cols_for_indicator:
        df_indicators[f'{col}_Is_Missing'] = df[col].isnull().astype(int)
    return df_indicators

def add_numeric_ratio_features(df_numeric):
    """
    Bổ sung các đặc trưng tỷ lệ giúp mô tả bố cục căn nhà rõ hơn.
    """
    df_enhanced = df_numeric.copy()

    floors = df_enhanced["Floors"].replace(0, np.nan)
    bedrooms = df_enhanced["Bedrooms"].replace(0, np.nan)
    area = df_enhanced["Area"].replace(0, np.nan)

    df_enhanced["Area_per_floor"] = df_enhanced["Area"] / floors
    df_enhanced["Rooms_per_floor"] = (df_enhanced["Bedrooms"] + df_enhanced["Bathrooms"]) / floors
    df_enhanced["Bathrooms_per_bedroom"] = df_enhanced["Bathrooms"] / bedrooms
    df_enhanced["Frontage_to_area"] = df_enhanced["Frontage"] / area
    df_enhanced["Bedrooms_to_area"] = df_enhanced["Bedrooms"] / area

    df_enhanced = df_enhanced.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_enhanced
