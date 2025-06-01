import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# === 資料來源設定 ===
data_dir = "TestData"
merged_files = glob.glob(os.path.join(data_dir, "Splited_Merged0*.csv"))
print(f"找到 {len(merged_files)} 個檔案：", merged_files)

processed_data = []

# === 處理每個檔案 ===
for file in merged_files:
    print(f"處理檔案: {file}")
    try:
        df = pd.read_csv(file)

        # 移除 inf、NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # === 🔍 移除無用欄位 ===
        drop_cols = ['Number', 'Tot sum', 'Tot size']
        df.drop(columns=drop_cols, errors='ignore', inplace=True)

        # Label 編碼
        le = LabelEncoder()
        df['Label'] = le.fit_transform(df['Label'])

        # 儲存 Label 對照表（只做一次）
        if not os.path.exists("label_map.csv"):
            label_map = dict(zip(le.classes_, le.transform(le.classes_)))
            pd.Series(label_map).to_csv("label_map.csv")
            print("✅ 已儲存 label_map.csv")

        # 標準化特徵欄位
        X = df.drop('Label', axis=1)
        y = df['Label']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        df_scaled['Label'] = y.values

        processed_data.append(df_scaled)

    except Exception as e:
        print(f"⚠️ 檔案 {file} 發生錯誤：{e}")

# === 合併所有處理後的資料 ===
final_df = pd.concat(processed_data, ignore_index=True)
print(f"合併後資料筆數：{len(final_df)}")

# === 🔎 篩選類別：刪除樣本數不足 3000，過多則下採樣到 50000 ===
label_counts = final_df['Label'].value_counts()
print("\n📊 原始類別分布：")
print(label_counts.sort_index())

# 只保留樣本數 ≥ 3000 的類別
valid_labels = label_counts[label_counts >= 3000].index
filtered_df = final_df[final_df['Label'].isin(valid_labels)].copy()

# 對樣本數 > 100000 的類別下採樣到 50000
downsampled_df = filtered_df.groupby('Label').apply(
    lambda x: x.sample(n=50000, random_state=42) if len(x) > 100000 else x
).reset_index(drop=True)

# 顯示處理後的類別分布
print("\n📊 處理後類別分布：")
print(downsampled_df['Label'].value_counts().sort_index())

# === 輸出結果 ===
downsampled_df.to_csv("processed_dataset_test.csv", index=False)
print("✅ 已儲存處理結果至 processed_dataset.csv")
