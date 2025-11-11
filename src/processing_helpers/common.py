import pandas as pd

def load_data(csv_path):
    """load dữ liệu"""
    try:
        df = pd.read_csv(csv_path)
        print("Tải dữ liệu thành công.")
        return df
    except FileNotFoundError:
        print(f"Không tìm thấy '{csv_path}'")
        return None
    
def drop_noisy_columns(df):
    "Loại bỏ các cột nhiễu/không dùng tới nữa"
    cols_to_drop = ['House direction', 'Balcony direction', 'Address']
    df = df.drop(columns=cols_to_drop)
    print(f"Đã loại bỏ các cột: {cols_to_drop}")
    return df