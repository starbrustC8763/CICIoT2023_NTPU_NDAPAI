# train_dnn.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

print("📥 讀取資料中...")
df = pd.read_csv("processed_dataset.csv")
print(f"✅ 資料載入完成，共 {df.shape[0]} 筆資料，{df.shape[1]} 欄位")

# 分離特徵與標籤
X = df.drop('Label', axis=1)
y_raw = df['Label']

le = joblib.load("label_encoder.pkl")
y = df['Label'].values.astype(int)  # 不再轉換，只強制為 int
num_classes = len(le.classes_)

# 儲存 label 對照表
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
pd.Series(label_map).to_csv("label_map.csv")
joblib.dump(le, "label_encoder.pkl")
print("🗂️ 已儲存 Label 對照表與編碼器")

# 特徵標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")
print("⚖️ 特徵已標準化並儲存")

# One-hot 編碼目標
y_cat = to_categorical(y, num_classes)

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.3, stratify=y, random_state=42)

# 建立 DNN 模型
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

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("🚀 開始訓練 DNN 模型（使用 GPU）...")
model.fit(X_train, y_train,
          epochs=20,
          batch_size=1024,
          validation_split=0.1,
          callbacks=[early_stop],
          verbose=2)

# 評估
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ 測試準確率: {acc:.4f}")

# 預測與報告
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


# 儲存模型
#model.save("dnn_model.h5")
#print("📦 模型已儲存為 dnn_model.h5")


# 類別名稱處理
target_names = [str(label) for label in le.classes_]
print("\n📊 分類報告：")
print(classification_report(
    y_true_classes, y_pred_classes,
    target_names=target_names,
    zero_division=0  # 避免因類別未預測出現錯誤
))

print(confusion_matrix(y_true_classes, y_pred_classes))