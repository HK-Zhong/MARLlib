import pandas as pd
import os


def convert_to_completion_ratio(input_file, output_file=None):
    # 读取数据
    df = pd.read_csv(input_file)

    # 自动识别列名（防止大小写问题）
    step_col = None
    value_col = None

    for col in df.columns:
        if col.lower() == "step":
            step_col = col
        if col.lower() == "value":
            value_col = col

    if step_col is None or value_col is None:
        raise ValueError("CSV中必须包含 Step 和 Value 列")

    # 只保留两列
    df_new = df[[step_col, value_col]].copy()

    # 计算 completion ratio（除以6）
    df_new[value_col] = df_new[value_col] / 6.0

    # 重命名列（统一格式）
    df_new.columns = ["Step", "Completion_Ratio"]

    # 输出文件名
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = base + "_completion_ratio_final.csv"

    # 保存
    df_new.to_csv(output_file, index=False)

    print(f"处理完成，保存为: {output_file}")


if __name__ == "__main__":
    convert_to_completion_ratio(
        input_file="src/datas/Ours_without_relational_encoder_extended.csv"  # 改成你的文件
    )