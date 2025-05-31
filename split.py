import pandas as pd
import glob
import os

# 讀取所有以 Merged 開頭的 CSV 檔案
merged_files = glob.glob("Merged*.csv")

# 對每個檔案進行切分
for file in merged_files:
    print(f"處理中：{file}")
    try:
        # 讀取原始檔案
        df = pd.read_csv(file)
        total_rows = len(df)
        chunk_size = total_rows // 4  # 每份大小（整除）

        base_name = os.path.splitext(file)[0]  # 去除副檔名（例如 Merged01）

        # 切成 4 份儲存
        for i in range(4):
            start = i * chunk_size
            # 最後一份包含剩下的所有資料
            end = (i + 1) * chunk_size if i < 3 else total_rows
            df_part = df.iloc[start:end]

            output_name = f"Splited_{base_name}_part{i+1}.csv"
            df_part.to_csv(output_name, index=False)
            print(f"✅ 已儲存：{output_name}（{len(df_part)} 筆）")

    except Exception as e:
        print(f"⚠️ 錯誤：{file} 無法處理，原因：{e}")
