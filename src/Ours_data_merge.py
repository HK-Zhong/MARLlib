import pandas as pd

# ===== 1. 读取数据 =====
file_new = "/home/coolas-fly/MARLlib/src/datas/exp_results_mappo_mlp_Scenario1_New.csv"
file_new2 = "/home/coolas-fly/MARLlib/src/datas/exp_results_mappo_mlp_Scenario1_New2.csv"

df_new = pd.read_csv(file_new)
df_new2 = pd.read_csv(file_new2)

# ===== 2. 确保按 Step 排序 =====
df_new = df_new.sort_values(by="Step")
df_new2 = df_new2.sort_values(by="Step")

# ===== 3. 分段筛选 =====
threshold = 650400

df_part1 = df_new[df_new["Step"] <= threshold]
df_part2 = df_new2[df_new2["Step"] > threshold]

# ===== 4. 合并 =====
df_merged = pd.concat([df_part1, df_part2], ignore_index=True)

# ===== 5. Value 截断（>6 → 6）=====
df_merged["Value"] = df_merged["Value"].clip(upper=6)

# ===== 6. 再排序（保险）=====
df_merged = df_merged.sort_values(by="Step")

# ===== 7. 保存 =====
output_file = "merged_results.csv"
df_merged.to_csv(output_file, index=False)

print(f"处理完成，保存为 {output_file}")