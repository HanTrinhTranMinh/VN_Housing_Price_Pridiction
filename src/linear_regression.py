# src/linear_regression.py
import numpy as np

def add_bias_column(X):
    """
    Thêm cột bias 1 vào đầu ma trận
    """
    X = np.asarray(X, dtype=np.float64)
    bias_column = np.ones((X.shape[0], 1))
    return np.c_[bias_column, X]

def normal_equation(X, y, lambda_reg=1e-2):
    """
    Tính vector theta theo normal equation có regularization (Ridge).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X_T = X.T
    X_T_X = X_T.dot(X)

    # Ridge regularization giúp ổn định nghịch đảo và giảm overfit.
    reg_matrix = np.eye(X_T_X.shape[0]) * lambda_reg
    reg_matrix[0, 0] = 0  # không regular hóa bias
    X_T_X_reg = X_T_X + reg_matrix

    try:
        X_T_X_inv = np.linalg.inv(X_T_X_reg)
    except np.linalg.LinAlgError:
        print("Ma trận X^T X suy biến, chuyển sang dùng pseudo-inverse.")
        X_T_X_inv = np.linalg.pinv(X_T_X_reg)

    theta = X_T_X_inv.dot(X_T.dot(y))
    return theta

def predict(X, theta):
    """dự đoán giá trị y"""
    X = np.asarray(X, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    return X.dot(theta)