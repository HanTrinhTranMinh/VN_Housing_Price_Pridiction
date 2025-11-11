# src/evaluate.py
import numpy as np

def mean_squared_error(y_actual, y_pred):
    """
    Tính Lỗi Bình phương Trung bình (Mean Squared Error - MSE)
    """
    error = y_pred - y_actual
    squared_error = error ** 2
    return np.mean(squared_error)

def root_mean_squared_error(y_actual, y_pred):
    """
    Tính Căn bậc hai của Lỗi Bình phương Trung bình (RMSE)
    """
    mse = mean_squared_error(y_actual, y_pred)
    return np.sqrt(mse)

def r_squared(y_actual, y_pred):
    """
    Tính Hệ số Xác định (R-squared)
    """
    ssr = np.sum((y_pred - y_actual) ** 2)
    y_mean = np.mean(y_actual)
    sst = np.sum((y_actual - y_mean) ** 2)
    if sst == 0:
        return 1.0
    return 1 - (ssr / sst)

# --- BỔ SUNG HÀM MỚI ---

def mean_absolute_error(y_actual, y_pred):
    """
    Tính Lỗi Tuyệt đối Trung bình (Mean Absolute Error - MAE)
    Công thức: (1/m) * Σ|y_pred - y_actual|
    """
    error = y_pred - y_actual
    absolute_error = np.abs(error)
    return np.mean(absolute_error)

def mean_absolute_scaled_error(y_actual, y_pred, y_train):
    """
    Tính Lỗi Tuyệt đối Trung bình Có co giãn (MASE)
    """
    # Lỗi của mô hình hiện tại trên tập test (Numerator)
    mae_model = mean_absolute_error(y_actual, y_pred)
    
    # Lỗi của mô hình "ngây thơ" trên tập train (Denominator)
    train_mean = np.mean(y_train)
    mae_naive_on_train = np.mean(np.abs(y_train - train_mean))
    
    # Xử lý trường hợp chia cho 0 (nếu y_train đều bằng nhau)
    if mae_naive_on_train == 0:
        return np.inf  # Lỗi vô hạn nếu không thể co giãn
        
    return mae_model / mae_naive_on_train