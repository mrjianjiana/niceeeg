# # import pandas as pd
# #
# # # df = pd.read_parquet("D:\master\brain\3.1\数据集\sub_01-00000-of-00001.parquet")
# # # 方法2：使用双反斜杠
# # # df = pd.read_parquet("D:\\master\\brain\\3.1\\数据集\\sub_01-00000-of-00001.parquet")
# #
# # # 方法3：使用正斜杠（跨平台兼容）
# # df = pd.read_parquet("D:/master/brain/3.1/数据集/sub_01-00000-of-00001.parquet")
# # print(df.head())  # 查看数据结构和标签
# import pandas as pd
# import numpy as np
#
# # 加载数据
# # df = pd.read_parquet(r"D:\master\brain\3.1\数据集\sub01-00000-of-00001.parquet")
# df = pd.read_parquet("D:/master/brain/3.1/数据集/sub_01-00000-of-00001.parquet")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast # 用于安全地解析字符串形式的字典（如果需要）

# 1. 加载数据
# file_path = "sub_01-00000-of-00001.parquet" # 替换为你的实际路径
#
# df = pd.read_parquet(file_path)
df = pd.read_parquet("D:/master/brain/3.1/数据集/sub_01-00000-of-00001.parquet")

print("===== 数据框基本信息 =====")
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print("\n===== 前几行数据 =====")
print(df.head())

# 2. 探索 eeg_array 列
print("\n===== 探索 EEG 数据 =====")
# 获取第一个样本的EEG数据
first_eeg = df.iloc[0]['eeg_array']
print(f"第一个样本的EEG数据类型: {type(first_eeg)}")
print(f"第一个样本的EEG数据形状: {first_eeg.shape}") # 这应该会输出类似 (64, 250) 的形状
print(f"EEG数据示例（第一个通道的前5个点）: {first_eeg[0, :5]}") # 查看一下数值

# 3. 探索 label 和 fold 列
print("\n===== 探索 Label 和 Fold =====")
print(f"'label' 的唯一值数量: {df['label'].nunique()}")
print(f"'label' 的值范围: {df['label'].min()} 到 {df['label'].max()}")
print(f"'fold' 的唯一值: {df['fold'].unique()}")

# 4. 探索 metadata 列 - 这是最关键的部分！
print("\n===== 探索 Metadata =====")
first_metadata = df.iloc[0]['metadata']
print(f"第一个样本的 metadata 类型: {type(first_metadata)}")
print(f"第一个样本的 metadata 内容: {first_metadata}")

# 提取 metadata 中的具体信息
# 查看所有 metadata 中都包含哪些键
all_keys = set()
for meta in df['metadata']:
    all_keys.update(meta.keys())
print(f"Metadata 中所有的键: {all_keys}")

# 提取出 'img_concept' 作为示例
img_concepts = [meta['img_concept'] for meta in df['metadata']]
print(f"前5个 image concepts: {img_concepts[:5]}")
print(f"唯一 image concepts 的数量: {len(set(img_concepts))}")

# 5. 可视化一个样本的一个通道
plt.figure(figsize=(12, 4))
plt.plot(first_eeg[0, :]) # 绘制第一个样本的第一个通道
plt.title(f"Sample 0, Channel 0 EEG Signal")
plt.xlabel("Time Points (250 Hz, 1 second)")
plt.ylabel("Amplitude (µV)")
plt.tight_layout()
plt.savefig('eeg_sample_plot.png')
plt.show()