import numpy as np
import pandas as pd
import re
def engineer_address_features(df):
    """
    Tách 'City' và 'District' từ 'Address' + chuẩn hóa tên tỉnh/thành và quận/huyện.
    Gồm các bước:
      - Xóa dấu chấm cuối
      - Chuẩn hóa viết tắt TP.HCM, HCM, Ho Chi Minh City → Ho Chi Minh
      - Chuẩn hóa HN, Hanoi → Ha Noi
      - Chuẩn hóa District (Q, Quan, H, Huyen…)
    """
    df['Address_clean'] = (
        df['Address']
        .astype(str)
        .str.strip()
        .str.replace(r'\.+$', '', regex=True)  # xoá dấu . cuối câu
    )


    address_parts = df['Address_clean'].str.split(r',\s*')

    df['City'] = address_parts.str[-1].fillna('Unknown')
    df['District'] = address_parts.str[-2].fillna('Unknown')

    
    city_map = {
        r'^(tp\.?\s*hcm|hcm|ho chi minh|ho chi minh city)$': "Ho Chi Minh",
        r'^(hn|ha noi|hanoi)$': "Ha Noi",
        r'^da nang$': "Da Nang",
        r'^can tho$': "Can Tho",
        r'^gia lai$': "Gia Lai"
    }

    def normalize_city(c):
        c = c.lower().strip().replace('.', '')

        for pattern, value in city_map.items():
            if re.match(pattern, c):
                return value

        return c.title()

    df['City'] = df['City'].apply(normalize_city)

    def normalize_district(d):
        d = d.lower().strip().replace('.', '')

        # chuẩn hóa quan
        d = re.sub(r'^(q|quan)\s*', 'Quan ', d)
        # chuẩn hóa huyen
        d = re.sub(r'^(h|huyen)\s*', 'Huyen ', d)
        # chuẩn hóa thi xa
        d = re.sub(r'^(tx|thixa)\s*', 'Thi Xa ', d)
        # tp
        d = re.sub(r'^(tp)\s*', 'TP ', d)

        return d.title()

    df['District'] = df['District'].apply(normalize_district)

    df = df.drop(columns=['Address_clean'], errors='ignore')
    print("✓ Hoàn tất chuẩn hóa Address, City, District.")
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
