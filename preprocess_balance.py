# preprocess_balanced.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# === 設定參數 ===
INPUT_FILE = "processed_dataset.csv"
OUTPUT_FILE = "processed_dataset_balanced.csv"
ENCODER_FILE = "label_encoder.pkl"
MIN_SAMPLES_PER_CLASS = 30  # 移除樣本數過少的類別

print(f"📥 載入資料檔案：{INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"原始資料筆數：{len(df)}")

# === 移除樣本數太少的類別 ===
label_counts = df['Label'].value_counts()
rare_classes = label_counts[label_counts < MIN_SAMPLES_PER_CLASS].index
df = df[~df['Label'].isin(rare_classes)]
print(f"✅ 已移除樣本 < {MIN_SAMPLES_PER_CLASS} 的類別，共移除 {len(rare_classes)} 類")

# === 分離特徵與標籤 ===
X = df.drop('Label', axis=1)
y = df['Label']

# === Label 編碼並儲存 ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, ENCODER_FILE)
print(f"✅ 已儲存 LabelEncoder 至 {ENCODER_FILE}")

# === 套用 SMOTE 過採樣 ===
print("🔁 套用 SMOTE 進行過採樣（平衡各類別）...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# === 還原為 DataFrame 並儲存 ===
X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled_df = pd.Series(y_resampled, name='Label')
df_balanced = pd.concat([X_resampled_df, y_resampled_df], axis=1)

# === 儲存 balanced 資料集 ===
df_balanced.to_csv(OUTPUT_FILE, index=False)
print(f"📦 平衡後資料儲存為 {OUTPUT_FILE}，總筆數：{len(df_balanced)}")
