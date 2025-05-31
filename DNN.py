import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# === STEP 1: 讀取資料集 ===
csv_path = "processed_dataset.csv"  # ← 換成你儲存的資料檔
df = pd.read_csv(csv_path)
print(f"✅ 載入成功：{df.shape[0]} 筆資料，{df.shape[1]} 欄位")

# === STEP 2: 特徵與標籤分離 ===
X = df.drop('Label', axis=1)
y_raw = df['Label']

# === STEP 3: Label 編碼（保留原始攻擊類型）===
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 儲存 label 對照表
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
pd.Series(label_map).to_csv("label_map.csv")
print("📋 Label 對應表已儲存")

# === STEP 4: 特徵標準化（如果尚未標準化）===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === STEP 5: One-hot 編碼標籤 ===
num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes)

# === STEP 6: 資料切分 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.3, stratify=y, random_state=42)

# === STEP 7: 建立 DNN 模型 ===
model = Sequential([
    Dense(256, input_dim=X.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 提早停止避免過度訓練
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# === STEP 8: 模型訓練 ===
model.fit(X_train, y_train,
          epochs=20,
          batch_size=1024,
          validation_split=0.1,
          callbacks=[early_stop])

# === STEP 9: 測試與評估 ===
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ 測試準確率: {accuracy:.4f}")

# === STEP 10: 預測與報告 ===
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\n📊 分類報告：")
print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

# === STEP 11: 儲存模型與編碼器 ===
model.save("dnn_model.h5")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ 模型與工具儲存完成（dnn_model.h5, label_encoder.pkl, scaler.pkl）")
