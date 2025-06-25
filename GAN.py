import pandas as pd
import numpy as np
from ctgan import CTGAN
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# 1. 读取数据
file_path = "C:/Users/Zhang/PAPER3.xlsx"
df = pd.read_excel(file_path)

# 如列名不匹配可注释下面一行
df.columns = ['qt', 'B', 'q', 'St']

# 2. 自动识别离散变量列
discrete_columns = []
for col in df.columns:
    if pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 15:
        discrete_columns.append(col)
print(f"识别为离散变量的列: {discrete_columns}")

# 3. 初始化CTGAN，训练
ctgan = CTGAN(epochs=2000, batch_size=100, verbose=True)
ctgan.fit(df, discrete_columns=discrete_columns)

# 4. 生成新数据
n_sample = 50
new_samples = ctgan.sample(n_sample)

# 5. 保存
new_samples.to_excel("PAPER3_synthetic_50.xlsx", index=False)

# 6. 评估：与原数据对比分布
def compare_distributions(real, synth, cols):
    for col in cols:
        plt.figure(figsize=(6,4))
        plt.title(f"Feature: {col} (real-blue; synth-orange)")
        plt.hist(real[col], bins=20, alpha=0.7, label='Real', density=True)
        plt.hist(synth[col], bins=20, alpha=0.7, label='Synth', density=True)
        plt.legend()
        plt.show()
        # KS检验
        D, pval = ks_2samp(real[col], synth[col])
        print(f"{col}   KS statistic={D:.3f}, p-value={pval:.3g}")

# 打印部分新数据
print(new_samples.head())

# 分布对比
compare_distributions(df, new_samples, df.columns)