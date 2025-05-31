import pandas as pd
import joblib

# === STEP 1: 載入模型與工具 ===
model = joblib.load("rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")  # 如果你前處理時有用 StandardScaler()

# === STEP 2: 載入測試資料 ===
# 這裡假設你要測試的資料格式跟訓練資料一樣
df = pd.read_csv("processed_dataset_test.csv")

# 分離特徵與標籤
X = df.drop("Label", axis=1)
y_true_raw = df["Label"]

# 標準化（如使用過）
X_scaled = scaler.transform(X)

# 將文字類別轉為數字（與訓練一致）
y_true = label_encoder.transform(y_true_raw)

# === STEP 3: 預測 ===
y_pred = model.predict(X)

# === STEP 4: 顯示結果 ===
from sklearn.metrics import classification_report, confusion_matrix

print("✅ 測試完成")
print("📊 混淆矩陣：")
print(confusion_matrix(y_true, y_pred))

print("\n📋 分類報告：")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
