import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === 檔案設定 ===
TEST_FILE = "processed_dataset_test.csv"
MODEL_FILE = "dnn_model_clean.h5"
SCALER_FILE = "scaler.pkl"
ENCODER_FILE = "label_encoder.pkl"
LABEL_MAP_FILE = "label_map.csv"
CONFUSION_IMG = "confusion_matrix_test_clean.png"

# === 讀取測試資料 ===
print(f"📥 載入測試資料：{TEST_FILE}")
df = pd.read_csv(TEST_FILE)

# 處理 Inf / NaN
df.replace([np.inf, -np.inf], pd.NA, inplace=True)
df.dropna(inplace=True)

# 刪除與訓練一致的欄位
drop_cols = ['Number', 'Tot sum', 'Tot size']
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# === 特徵與標籤分離 ===
X = df.drop('Label', axis=1)
y_raw = df['Label']

# === 載入前處理工具 ===
scaler = joblib.load(SCALER_FILE)
le = joblib.load(ENCODER_FILE)
model = load_model(MODEL_FILE)

# === 套用 scaler（不能重新 fit）===
X_scaled = scaler.transform(X)

# === 標籤轉換（數字）===
y_true = le.transform(y_raw)

# === 預測 ===
y_pred_prob = model.predict(X_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

# === 還原 Label Map（從 CSV）===
label_map_df = pd.read_csv(LABEL_MAP_FILE, index_col=0, header=None)
id_to_label = {v: k for k, v in label_map_df[1].items()}
target_names = [id_to_label[i] for i in sorted(set(y_true) | set(y_pred))]

# === 分類報告 ===
print("\n📊 分類報告：")
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

# === 混淆矩陣圖 ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(18, 14))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(CONFUSION_IMG)
print(f"🖼️ 混淆矩陣已儲存為 {CONFUSION_IMG}")
