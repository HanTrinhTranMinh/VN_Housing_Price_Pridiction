# main.py
import pandas as pd
import numpy as np
from src.data_processing import load_and_preprocess_data
from src.linear_regression import add_bias_column, normal_equation, predict
from sklearn.linear_model import LinearRegression
from src.evaluate import (
    mean_squared_error, 
    root_mean_squared_error, 
    r_squared,
    mean_absolute_error, 
    mean_absolute_scaled_error 
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import  XGBRegressor
import matplotlib.pyplot as plt


def tune_lambda_with_kfold(X, y_transformed, y_orig, lambda_values, inverse_fn, n_splits=5):
    """
    Chọn lambda tốt nhất bằng K-Fold CV trên dữ liệu train, đánh giá trên thang đo gốc.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_lambda = None
    best_rmse = float("inf")

    for lam in lambda_values:
        fold_rmses = []
        print(f"Đang thử nghiệm với lambda = {lam}")
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr_transformed = y_transformed[train_idx]
            y_val_orig = y_orig[val_idx]

            theta = normal_equation(X_tr, y_tr_transformed, lambda_reg=lam)
            if theta is None:
                continue

            y_val_pred_transformed = predict(X_val, theta)
            y_val_pred = inverse_fn(y_val_pred_transformed)
            rmse = root_mean_squared_error(y_val_orig, y_val_pred)
            fold_rmses.append(rmse)

        if not fold_rmses:
            print("  Không thể tính RMSE cho lambda này, bỏ qua.")
            continue

        avg_rmse = float(np.mean(fold_rmses))
        print(f"  -> RMSE trung bình (CV): {avg_rmse:.4f}")

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_lambda = lam
            print("  *** Tìm thấy lambda tốt hơn thông qua CV! ***")

    return best_lambda, best_rmse


def inverse_log_transform(y_log_values, clip_min=-20, clip_max=20):
    """
    Chuyển các giá trị log1p về thang đo gốc với clipping để tránh overflow.
    """
    clipped = np.clip(y_log_values, clip_min, clip_max)
    return np.expm1(clipped)


def identity_transform(values, *_args, **_kwargs):
    """Hàm giữ nguyên giá trị (dùng cho biến thể target gốc)."""
    return values

def run_training_pipeline():
    
    DATA_PATH = "data/vietnam_housing_dataset.csv" # Sửa lại tên file nếu cần
    print("--- BƯỚC 1: TẢI VÀ XỬ LÝ DỮ LIỆU ---")
    X, y, feature_names = load_and_preprocess_data(DATA_PATH)
    
    if X is None or y is None:
        print("Kết thúc do lỗi tải dữ liệu.")
        return

    print("\n--- BƯỚC 2: THÊM CỘT BIAS ---")
    X_b = add_bias_column(X)
    print(f"Tổng số mẫu: {X_b.shape[0]}, Tổng số đặc trưng (có bias): {X_b.shape[1]}")

    print("\n--- BƯỚC 3: CHIA DỮ LIỆU (TRAIN/TEST) ---")
    y_log = np.log1p(np.maximum(y, 0))
    (
        X_train,
        X_test,
        y_train_log,
        _y_test_log,
        y_train_orig,
        y_test_orig,
    ) = train_test_split(X_b, y_log, y, test_size=0.2, random_state=42)
    print(f"Kích thước tập Train: {X_train.shape[0]} mẫu")
    print(f"Kích thước tập Test: {X_test.shape[0]} mẫu")

    # --- BƯỚC 4: TINH CHỈNH SIÊU THAM SỐ (lambda) ---
    print("\n--- BƯỚC 4: TINH CHỈNH SIÊU THAM SỐ (lambda) ---")
    
    lambda_values_to_try = [0.001, 0.01, 0.1, 1, 10, 100]
    target_variants = [
        {
            "name": "log1p(target)",
            "y_train": y_train_log,
            "inverse_fn": inverse_log_transform,
        },
        {
            "name": "target gốc",
            "y_train": y_train_orig,
            "inverse_fn": identity_transform,
        },
    ]

    best_variant = None
    for variant in target_variants:
        print(f"\n>>> Đánh giá biến thể target: {variant['name']}")
        lam, cv_rmse = tune_lambda_with_kfold(
            X_train,
            variant["y_train"],
            y_train_orig,
            lambda_values_to_try,
            variant["inverse_fn"],
        )
        if lam is None:
            continue

        if (best_variant is None) or (cv_rmse < best_variant["cv_rmse"]):
            best_variant = {
                "name": variant["name"],
                "lambda": lam,
                "cv_rmse": cv_rmse,
                "y_train": variant["y_train"],
                "inverse_fn": variant["inverse_fn"],
            }

    if best_variant is None:
        print("Không tìm được cấu hình target nào phù hợp, kết thúc pipeline.")
        return

    best_lambda = best_variant["lambda"]
    inverse_target_fn = best_variant["inverse_fn"]
    y_train_for_fit = best_variant["y_train"]

    print("\n--- KẾT THÚC TINH CHỈNH ---")
    print(
        f"==> Cấu hình: {best_variant['name']} với lambda = {best_lambda} "
        f"(RMSE trung bình CV = {best_variant['cv_rmse']:.4f})"
    )

    theta = normal_equation(X_train, y_train_for_fit, lambda_reg=best_lambda)
    
    if theta is None:
        print("Kết thúc do lỗi huấn luyện (không tìm thấy theta nào hợp lệ).")
        return
        
    print("Huấn luyện thành công với lambda tốt nhất.")


    print("\n--- BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH TỰ TẠO TỐT NHẤT (TRÊN TẬP TEST) ---")
    
    # Tạo dự đoán cuối cùng bằng mô hình tốt nhất
    y_pred_transformed = predict(X_test, theta)
    y_pred = inverse_target_fn(y_pred_transformed)
    
    # Tính toán tất cả các chỉ số
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = root_mean_squared_error(y_test_orig, y_pred)
    r2 = r_squared(y_test_orig, y_pred)    
    mae = mean_absolute_error(y_test_orig, y_pred)
    mase = mean_absolute_scaled_error(y_test_orig, y_pred, y_train_orig) 
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CUỐI CÙNG ---")
    print(f" (Biến thể target: {best_variant['name']} | lambda = {best_lambda})")
    print(f"  R-squared (Hệ số Xác định):     {r2:.4f} (hoặc {r2*100:.2f}%)")
    print(f"  MSE (Lỗi Bình phương Trung bình): {mse:.4f}")
    print(f"  RMSE (Lỗi Trung bình):           {rmse:.4f}")
    print(f"  MAE (Lỗi Tuyệt đối Trung bình):  {mae:.4f}")  
    print(f"  MASE (Lỗi Co giãn Trung bình):   {mase:.4f}") 

    # VẼ SƠ ĐỒ
    plt.figure(figsize=(8, 6))

# Vẽ scatter điểm dự đoán
    plt.scatter(y_test_orig, y_pred, alpha=0.7)

    # Vẽ đường tham chiếu y = x
    min_val = min(min(y_test_orig), min(y_pred))
    max_val = max(max(y_test_orig), max(y_pred))

    plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)

    plt.xlabel("Giá trị thật (y_test)")
    plt.ylabel("Giá trị dự đoán (y_pred)")
    plt.title("So sánh Giá trị Thật vs Dự đoán (Prediction Plot)")
    plt.grid(True)

    plt.show()
 

    # So sánh với Linear Regression từ sklearn
    lr = LinearRegression()
    lr.fit(X_train, y_train_for_fit)   # 👉 dùng đúng target đã dùng cho mô hình tự tạo

    lr_pred_transformed = lr.predict(X_test)
    lr_pred = inverse_target_fn(lr_pred_transformed)

    # -----------------------------------------------------
    # 🔍 TÍNH TOÁN CHỈ SỐ CHO MÔ HÌNH LINEAR REGRESSION (sklearn)
    # -----------------------------------------------------
    lr_mse = mean_squared_error(y_test_orig, lr_pred)
    lr_rmse = root_mean_squared_error(y_test_orig, lr_pred)
    lr_r2 = r_squared(y_test_orig, lr_pred)
    lr_mae = mean_absolute_error(y_test_orig, lr_pred)
    lr_mase = mean_absolute_scaled_error(y_test_orig, lr_pred, y_train_orig)

    print("\n--- KẾT QUẢ MÔ HÌNH LINEAR REGRESSION (sklearn) ---")
    print(f"  R-squared:                      {lr_r2:.4f}")
    print(f"  MSE:                            {lr_mse:.4f}")
    print(f"  RMSE:                           {lr_rmse:.4f}")
    print(f"  MAE:                            {lr_mae:.4f}")
    print(f"  MASE:                           {lr_mase:.4f}")

    # =====================================================
    # 🎯 ĐỘ TRÙNG KHỚP GIỮA MÔ HÌNH TỰ TẠO VÀ SKLEARN
    # =====================================================
    diff = np.abs(y_pred - lr_pred)
    avg_diff = diff.mean()
    max_diff = diff.max()
    min_diff = diff.min()

    print("\n=========== ĐỘ TRÙNG KHỚP GIỮA HAI MÔ HÌNH ===========")
    print(f"Chênh lệch trung bình (|y_self - y_sklearn|):  {avg_diff:.4f}")
    print(f"Chênh lệch lớn nhất:                           {max_diff:.4f}")
    print(f"Chênh lệch nhỏ nhất:                           {min_diff:.4f}")


    # =====================================================
    # 🎯 XGBOOST REGRESSOR — SO SÁNH MẠNH NHẤT
    # =====================================================
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )


    # Huấn luyện XGBoost (dùng target gốc)
    xgb.fit(X_train, y_train_orig)

    # Dự đoán
    xgb_pred = xgb.predict(X_test)

    # Tính toán các chỉ số
    xgb_mse = mean_squared_error(y_test_orig, xgb_pred)
    xgb_rmse = root_mean_squared_error(y_test_orig, xgb_pred)
    xgb_r2 = r_squared(y_test_orig, xgb_pred)
    xgb_mae = mean_absolute_error(y_test_orig, xgb_pred)
    xgb_mase = mean_absolute_scaled_error(y_test_orig, xgb_pred, y_train_orig)

    print("\n--- KẾT QUẢ MÔ HÌNH XGBOOST REGRESSOR ---")
    print(f"  R-squared: {xgb_r2:.4f}")
    print(f"  MSE:       {xgb_mse:.4f}")
    print(f"  RMSE:      {xgb_rmse:.4f}")
    print(f"  MAE:       {xgb_mae:.4f}")
    print(f"  MASE:      {xgb_mase:.4f}")

if __name__ == "__main__":
    run_training_pipeline()
