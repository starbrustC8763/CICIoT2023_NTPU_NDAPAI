# 使用 TensorFlow GPU 版作為基底
FROM tensorflow/tensorflow:2.13.0-gpu

# 工作目錄（對應掛載的資料夾）
WORKDIR /workspace

# 安裝你需要的套件
RUN pip install --no-cache-dir pandas scikit-learn joblib seaborn matplotlib imblearn

# 設定預設執行指令（可改為 bash 以進入容器）
CMD ["bash"]

