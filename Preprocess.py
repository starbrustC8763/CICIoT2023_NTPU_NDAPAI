import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler,LabelEncoder
data_dir = "Data"  # 資料夾名稱
merged_files = glob.glob(os.path.join(data_dir, "Splited_Merged0*.csv"))
# === STEP 1: 讀取多個 Merged CSV 檔案 === 
print(f"找到 {len(merged_files)} 個檔案：", merged_files)

# 儲存所有處理過的資料
processed_data = []

# === STEP 2: 逐一處理每個檔案 ===
for file in merged_files:
    print(f"處理檔案: {file}")
    
    try:
        df = pd.read_csv(file)

        # 處理 inf/-inf 和 NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # 將 Label 轉成 0/1（二分類：Benign vs Attack）
        le = LabelEncoder()
        df['Label'] = le.fit_transform(df['Label'])

        # 儲存 label 編碼對照表
        label_map = dict(zip(le.classes_, le.transform(le.classes_)))
        pd.Series(label_map).to_csv("label_map.csv")
        # 特徵與標籤分離
        X = df.drop('Label', axis=1)
        y = df['Label']

        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 重建標準化後的 DataFrame 並加回 Label
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        X_scaled_df['Label'] = y.values

        # 加入總表
        processed_data.append(X_scaled_df)

    except Exception as e:
        print(f"⚠️ 檔案 {file} 發生錯誤：{e}")

# === STEP 3: 合併所有處理後的資料 ===
final_df = pd.concat(processed_data, ignore_index=True)
print("合併後資料筆數：", len(final_df))

# === STEP 4: 儲存為新的 CSV 檔案 ===
final_df.to_csv("processed_dataset.csv", index=False)
print("✅ 已儲存至 processed_dataset.csv")
