import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 读取数据
file_path = 'C:/Users/Zhang/PAPER3.xlsx'
df = pd.read_excel(file_path)

# 提取自变量和因变量
X = df[['B', 'q', 'qt']]
y = df['St']

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用固定的测试集比例 0.1 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# 创建并训练随机森林模型
model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2,
                              min_samples_leaf=1, max_features='sqrt', random_state=42)
model.fit(X_train, y_train)

# 对整个数据集进行预测
y_pred_all = model.predict(X_scaled)

# 评估模型性能
r2_all = r2_score(y, y_pred_all)
rmse_all = np.sqrt(mean_squared_error(y, y_pred_all))
mad_all = mean_absolute_error(y, y_pred_all)
mape_all = np.mean(np.abs((y - y_pred_all) / y)) * 100

print("R²:", r2_all)
print("RMSE:", rmse_all)
print("MAD:", mad_all)
print("MAPE:", mape_all)

# 可视化整个数据集的实际值和预测值
indices = np.arange(len(y))
plt.figure(figsize=(10, 6))
plt.scatter(indices, y, color="blue", label="Actual Values")
plt.plot(indices, y, color="blue", linestyle="-")
plt.scatter(indices, y_pred_all, color="red", label="Predicted Values")
plt.plot(indices, y_pred_all, color="red", linestyle="--")

# 在右上角添加模型评估指标
textstr = '\n'.join((
    f'R²: {r2_all:.3f}',
    f'RMSE: {rmse_all:.3f}',
    f'MAD: {mad_all:.3f}',
    f'MAPE: {mape_all:.2f}%'))

props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.15, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

plt.title("Comparison between the actual values and predicted values of St using the RF model")
plt.xlabel("Data number")
plt.ylabel("St(mm)")
plt.legend()
plt.show()