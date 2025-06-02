# train_dnn_clean.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# === 參數 ===
INPUT_FILE = "processed_dataset.csv"
MODEL_FILE = "dnn_model_clean.h5"
ENCODER_FILE = "label_encoder.pkl"
SCALER_FILE = "scaler.pkl"
CONFUSION_IMG = "confusion_matrix_clean.png"

# === 讀取資料 ===
print(f"📥 載入資料：{INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# === 特徵與標籤分離 ===
X = df.drop("Label", axis=1)
y_raw = df["Label"]

# === LabelEncoder（還原類別名用） ===
le = LabelEncoder()
y = le.fit_transform(y_raw)
joblib.dump(le, ENCODER_FILE)

# === 標準化特徵（此時為保險起見再次做標準化）===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_FILE)

# === One-hot 編碼標籤 ===
num_classes = len(le.classes_)
y_cat = to_categorical(y, num_classes)

# === 分割訓練/測試集 ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.3, stratify=y, random_state=42)

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
print("🚀 開始訓練模型...")
history = model.fit(X_train, y_train,
          epochs=20,
          batch_size=1024,
          validation_split=0.1,
          callbacks=[early_stop],
          verbose=2)

# === 模型評估 ===
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ 測試準確率: {acc:.4f}")

# === 預測與分類報告 ===
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

target_names = [str(cls) for cls in le.classes_]
print("\n📊 分類報告：")
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

# === 混淆矩陣可視化 ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(18, 14))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix (Cleaned Data)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(CONFUSION_IMG)
print(f"🖼️ 混淆矩陣儲存為 {CONFUSION_IMG}")

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
print("📈 已儲存 loss 曲線圖為 loss_curve.png")

# === 儲存模型 ===
model.save(MODEL_FILE)
print(f"✅ 模型已儲存為 {MODEL_FILE}")
