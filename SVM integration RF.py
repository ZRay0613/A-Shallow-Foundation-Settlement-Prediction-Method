import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import shap

# 创建自定义集成模型
class SVRRFModel(BaseEstimator, RegressorMixin):
    def __init__(self, svr_kernel='rbf', svr_C=1.0, svr_epsilon=0.1, rf_estimators=100, max_features='sqrt'):
        self.svr_kernel = svr_kernel
        self.svr_C = svr_C
        self.svr_epsilon = svr_epsilon
        self.rf_estimators = rf_estimators
        self.max_features = max_features

        self.svr = SVR(kernel=self.svr_kernel, C=self.svr_C, epsilon=self.svr_epsilon)
        self.rf = RandomForestRegressor(n_estimators=self.rf_estimators, max_features=self.max_features, random_state=42)

    def fit(self, X, y):
        svr_pred = self.svr.fit(X, y).predict(X)
        self.rf.fit(svr_pred.reshape(-1, 1), y)
        return self

    def predict(self, X):
        svr_pred = self.svr.predict(X)
        return self.rf.predict(svr_pred.reshape(-1, 1))

# 读取数据
file_path = 'C:/Users/Zhang/PAPER3.xlsx'
df = pd.read_excel(file_path)

# 提取自变量和因变量
X = df[['B', 'q', 'qt']]
y = df['St']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用PCA降低维度或增加特征
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 设置用于交叉验证的折数
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 实例化模型
model = SVRRFModel()

# 设置更广泛的参数网格
param_grid = {
    'svr_C': [0.01, 0.1, 1, 10, 100],
    'svr_epsilon': [0.001, 0.01, 0.1, 1],
    'rf_estimators': [50, 100, 200, 300]
}

# 使用网格搜索和交叉验证优化超参数
grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='r2', n_jobs=-1)
grid_search.fit(X_pca, y)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数和比例划分数据集
best_model = grid_search.best_estimator_

# 寻找最佳训练集和测试集比例
best_split_ratio = 0.1
best_score = -np.inf

for test_size in [0.05, 0.1, 0.15, 0.2]:
    scores = []
    for train_idx, test_idx in kf.split(X_pca):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)

    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_split_ratio = test_size

print("Best split ratio found: ", best_split_ratio)

# 使用最佳参数和比例划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=best_split_ratio, random_state=42)
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# 预测整个数据集
y_pred_full = best_model.predict(X_pca)

# 单独预测测试数据集用于评估性能
y_pred_test = best_model.predict(X_test)

# 评估模型性能
r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mad = mean_absolute_error(y_test, y_pred_test)
mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print("R²:", r2)
print("RMSE:", rmse)
print("MAD:", mad)
print("MAPE:", mape)


# 可视化训练和测试集中的实际值和预测值
indices = np.arange(len(y))
plt.figure(figsize=(10, 6))
plt.scatter(indices, y, color="blue", label="Actual Values")
plt.plot(indices, y, color="blue", linestyle="-")
plt.scatter(indices, y_pred_full, color="red", label="Predicted Values")
plt.plot(indices, y_pred_full, color="red", linestyle="--")

# 在右上角添加模型评估指标
textstr = '\n'.join((
    f'R²: {r2:.3f}',
    f'RMSE: {rmse:.3f}',
    f'MAD: {mad:.3f}',
    f'MAPE: {mape:.2f}%'))

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.15, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.title("Comparison between the actual and predicted values of St using integrated model")
plt.xlabel("Data number")
plt.ylabel("St(mm)")
plt.legend()
plt.show()


# 可视化预测值和实际值
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_test, edgecolors=(0, 0, 0), label='Predicted Values')
plt.plot([0, 300], [0, 300], 'k--', lw=2, label='Ideal situation')
plt.xlabel('Actual Values (mm)', fontsize=16)
plt.ylabel('Predicted Values (mm)', fontsize=16)
plt.title('The SVM integrated RF model', fontsize=16)
plt.legend(fontsize=12)
plt.grid()
# 设置X轴和Y轴的刻度
plt.xticks(np.arange(0, 301, 50))
plt.yticks(np.arange(0, 301, 50))
plt.text(0.76, 0.05, f"R²: {r2:.3f}\nRMSE: {rmse:.3f}\nMAPE: {mape:.3f}%\nMAD: {mad:.3f}", fontsize=16,
         transform=plt.gca().transAxes, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
plt.xlim(0, 300)
plt.ylim(0, 300)
plt.show()

# 使用完整数据集进行预测
y_pred_full = best_model.predict(X_pca)

# 可视化完整数据集的预测值和实际值
plt.figure(figsize=(10, 8))
plt.scatter(y, y_pred_full, edgecolors=(0, 0, 0), label='Predicted Values')
plt.plot([0, 300], [0, 300], 'k--', lw=2, label='Ideal situation')
plt.xlabel('Actual Values (mm)', fontsize=16)
plt.ylabel('Predicted Values (mm)', fontsize=16)
plt.title('The SVM integrated RF model', fontsize=16)
plt.legend(fontsize=12)
plt.grid()
# 设置X轴和Y轴的刻度
plt.xticks(np.arange(0, 301, 50))
plt.yticks(np.arange(0, 301, 50))
plt.text(0.76, 0.05, f"R²: {r2:.3f}\nRMSE: {rmse:.3f}\nMAPE: {mape:.3f}%\nMAD: {mad:.3f}", fontsize=16,
         transform=plt.gca().transAxes, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
plt.xlim(0, 300)
plt.ylim(0, 300)
plt.show()

# Sensitivity analysis
def sensitivity_analysis(base_model, X, original_X, feature_names):
    baseline_prediction = base_model.predict(X)
    sensitivity = pd.DataFrame(index=feature_names, columns=["sensitivity"])

    for i, feature in enumerate(feature_names):
        perturbed_X = original_X.copy()
        perturbed_X[feature] += 0.1 * np.std(perturbed_X[feature])  # 增加一个标准差的10%
        perturbed_X_scaled = scaler.transform(perturbed_X)  # Apply scaling
        perturbed_X_pca = pca.transform(perturbed_X_scaled)  # Apply PCA
        perturbed_prediction = base_model.predict(perturbed_X_pca)
        sensitivity.loc[feature] = np.mean(np.abs(perturbed_prediction - baseline_prediction))

    return sensitivity.sort_values(by="sensitivity", ascending=False)


feature_names = ['B', 'q', 'qt']
sensitivity = sensitivity_analysis(best_model, X_pca, X, feature_names)
print("\nSensitivity Analysis:")
print(sensitivity)

# Plotting the sensitivity analysis results
def plot_sensitivity(sensitivity):
    plt.figure(figsize=(8, 6))
    sensitivity_values = sensitivity["sensitivity"].astype(float)
    sensitivity_values.plot(kind='bar', color='skyblue', edgecolor='black')

    # Add sensitivity values on top of the bars
    for i, value in enumerate(sensitivity_values):
        plt.text(i, value + 0.01, f"{value:.4f}", ha='center', va='bottom', fontsize=10)

    plt.title("Sensitivity Analysis", fontsize=16)
    plt.ylabel("Sensitivity", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_sensitivity(sensitivity)

# Visualize sensitivity analysis
plt.figure(figsize=(8, 6))
plt.bar(sensitivity.index, sensitivity['sensitivity'])
plt.title("Feature Sensitivity Analysis")
plt.xlabel("Feature")
plt.ylabel("Sensitivity")
plt.show()

# Sensitivity analysis function
def plot_sensitivity_analysis(model, X, original_X, feature_name, scaler, pca):
    baseline_prediction = model.predict(X)
    percentage_changes = np.linspace(-0.1, 0.1, 5)  # -10% to 10%
    mean_changes = []

    for change in percentage_changes:
        perturbed_X = original_X.copy()
        perturbed_X[feature_name] += change * np.std(perturbed_X[feature_name])
        perturbed_X_scaled = scaler.transform(perturbed_X)
        perturbed_X_pca = pca.transform(perturbed_X_scaled)
        perturbed_prediction = model.predict(perturbed_X_pca)
        mean_change = np.mean(perturbed_prediction - baseline_prediction)
        mean_changes.append(mean_change)

    plt.figure(figsize=(8, 6))
    plt.plot(percentage_changes * 100, mean_changes, marker='o')
    plt.title(f"Sensitivity Analysis for {feature_name}")
    plt.xlabel("Percentage Change in Input Variable")
    plt.ylabel("Mean Change in Predicted St")
    plt.grid(True)
    plt.show()

# 对每个特征进行敏感性分析
feature_names = ['B', 'q', 'qt']
for feature in feature_names:
    plot_sensitivity_analysis(best_model, X_pca, X, feature, scaler, pca)

# SHAP分析
explainer = shap.Explainer(best_model.predict, X_pca)
shap_values = explainer(X_pca)

# SHAP总结图
shap.summary_plot(shap_values, X_pca, feature_names=['B', 'q', 'qt'])
