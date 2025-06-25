import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt


# 自定义SVR模型类
class SVRModel(BaseEstimator, RegressorMixin):
    def __init__(self, svr_kernel='rbf', svr_degree=2, poly_degree=2, C=1.0, epsilon=0.1, gamma='scale'):
        self.svr_kernel = svr_kernel
        self.svr_degree = svr_degree
        self.poly_degree = poly_degree
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma

        self.svr = SVR(kernel=self.svr_kernel, degree=self.svr_degree, C=self.C, epsilon=self.epsilon, gamma=self.gamma)
        self.poly = PolynomialFeatures(degree=self.poly_degree)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.svr.fit(X_poly, y)
        return self

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.svr.predict(X_poly)

    def get_params(self, deep=True):
        return {"svr_kernel": self.svr_kernel, "svr_degree": self.svr_degree,
                "poly_degree": self.poly_degree, "C": self.C, "epsilon": self.epsilon, "gamma": self.gamma}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.svr = SVR(kernel=self.svr_kernel, degree=self.svr_degree, C=self.C, epsilon=self.epsilon, gamma=self.gamma)
        self.poly = PolynomialFeatures(degree=self.poly_degree)
        return self

    # 读取数据


file_path = 'C:/Users/Zhang/PAPER3.xlsx'
df = pd.read_excel(file_path)

# 提取自变量和因变量
X = df[['B', 'q', 'qt']]
y = df['St']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用交叉验证寻找最佳测试集比例
best_score = float('inf')
best_test_size = None

k_values = [5, 10, 15, 20, 25, 30]

for k in k_values:
    test_size = k / 100
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    model = SVRModel()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Hyperparameter optimization using GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 1],
        'svr_kernel': ['linear', 'poly', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # Evaluate with the best params
    model.set_params(**best_params)
    scores = -cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)
    avg_score = np.mean(scores)

    if avg_score < best_score:
        best_score = avg_score
        best_test_size = test_size

print(f"Best test size: {best_test_size}, with MSE: {best_score}")

# 使用最佳比例划分数据，并训练模型
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=best_test_size, random_state=42)

model = SVRModel(**best_params)
model.fit(X_train, y_train)

# 对整个数据集进行预测
y_pred = model.predict(X_scaled)

# 评估模型性能
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mad = mean_absolute_error(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100

print("R²:", r2)
print("RMSE:", rmse)
print("MAD:", mad)
print("MAPE:", mape)

# 可视化所有点的实际值和预测值
indices = np.arange(len(y))
plt.figure(figsize=(10, 6))
plt.scatter(indices, y, color="blue", label="Actual Values")
plt.plot(indices, y, color="blue", linestyle="-")
plt.scatter(indices, y_pred, color="red", label="Predicted Values")
plt.plot(indices, y_pred, color="red", linestyle="--")

# 在右上角添加模型评估指标
textstr = '\n'.join((
    f'R²: {r2:.3f}',
    f'RMSE: {rmse:.3f}',
    f'MAD: {mad:.3f}',
    f'MAPE: {mape:.2f}%'))

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.15, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.title("Comparison between the actual values and predicted values of St using the SVM model")
plt.xlabel("Data number")
plt.ylabel("St(mm)")
plt.legend()
plt.show()