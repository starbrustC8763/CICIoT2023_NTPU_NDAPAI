import pandas as pd

# 載入預處理後的 CSV（請視情況修改檔名）
df = pd.read_csv("processed_dataset.csv")

# 統計 Label 欄位的類別數量
label_counts = df['Label'].value_counts()

# 顯示每個類別的筆數
print("📊 各類別樣本數統計：")
print(label_counts)

# 如果你想排序顯示（由小到大）
print("\n📊 類別樣本數（由少到多）：")
print(label_counts.sort_values())
