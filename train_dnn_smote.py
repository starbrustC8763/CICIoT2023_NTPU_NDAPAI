# train_dnn_smote.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# === 參數設定 ===
DATA_FILE = "processed_dataset.csv"  # 原始資料（不平衡）
MODEL_FILE = "dnn_model_smote.h5"
LABEL_ENCODER_FILE = "label_encoder.pkl"
SCALER_FILE = "scaler.pkl"
CONFUSION_MATRIX_PNG = "confusion_matrix_smote.png"

# === 載入資料 ===
print(f"📥 載入原始資料：{DATA_FILE}")
df = pd.read_csv(DATA_FILE)
X = df.drop('Label', axis=1)
y_raw = df['Label']

# === Label 編碼 ===
le = LabelEncoder()
y = le.fit_transform(y_raw)
joblib.dump(le, LABEL_ENCODER_FILE)
print("✅ LabelEncoder 已儲存")

# === 切分訓練與測試資料（測試集保持真實分布）===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# === 特徵標準化器（fit 於訓練集）===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, SCALER_FILE)
print("✅ 標準化器已儲存")

# === 對訓練集做 SMOTE ===
print("🔁 套用 SMOTE 於訓練集...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"訓練集樣本數：{len(X_train_res)}（原：{len(X_train)}）")

# === One-hot 編碼 ===
num_classes = len(le.classes_)
y_train_cat = to_categorical(y_train_res, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# === 建立 DNN 模型 ===
model = Sequential([
    Dense(256, input_dim=X.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 訓練模型 ===
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print("🚀 開始訓練模型（使用 SMOTE 平衡後的訓練集）...")
model.fit(X_train_res, y_train_cat,
          epochs=20,
          batch_size=1024,
          validation_split=0.1,
          callbacks=[early_stop],
          verbose=2)

# === 測試與分類報告 ===
y_pred_prob = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred_prob, axis=1)

target_names = [str(name) for name in le.classes_]
print("\n📊 測試集分類報告：")
print(classification_report(y_test, y_pred_classes, target_names=target_names, zero_division=0))

# === 混淆矩陣圖 ===
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(18, 14))
sns.heatmap(cm, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix (Test Set, SMOTE Training)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PNG)
print(f"🖼️ 測試混淆矩陣已儲存為 {CONFUSION_MATRIX_PNG}")

# === 儲存模型 ===
model.save(MODEL_FILE)
print(f"✅ 模型已儲存為 {MODEL_FILE}")
