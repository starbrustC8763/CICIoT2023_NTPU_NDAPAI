import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# === STEP 1: 載入資料集 ===
csv_path = "processed_dataset.csv"  # ← 換成你的資料檔路徑
df = pd.read_csv(csv_path)
print(f"✅ 載入資料成功：共 {df.shape[0]} 筆資料，{df.shape[1]} 個欄位")

# === STEP 2: 切分特徵與標籤 ===
X = df.drop('Label', axis=1)
y_raw = df['Label']

# === STEP 3: 對 Label 進行編碼（多分類）===
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 可選：儲存標籤對照表
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print("\n📋 類別對照表:")
for idx, label in enumerate(le.classes_):
    print(f"{idx}: {label}")
pd.Series(label_map).to_csv("label_map.csv")

# === STEP 4: 切分訓練與測試集 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# === STEP 5: 訓練 Random Forest 模型 ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# === STEP 7: 儲存模型（可選） ===
joblib.dump(clf, "rf_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ 模型與標籤編碼器已儲存。")

# === STEP 6: 預測與評估 ===
y_pred = clf.predict(X_test)

print("\n=== 混淆矩陣 ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== 分類報告 ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))


