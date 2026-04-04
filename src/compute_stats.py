import pandas as pd
import numpy as np
import os


def find_column(df, candidates):
    """自动识别列名（防止 Step / step / VALUE 等问题）"""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"未找到列名，候选列为: {candidates}")


def compute_value_stats(input_file, use_div7=False):
    """
    计算 CSV 中 Value 列的均值和方差（用于论文表格）
    """

    # 1. 读取数据
    df = pd.read_csv(input_file)

    # 2. 找列
    value_col = find_column(df, ["Value", "value", "VALUE"])

    # 3. 计算统计量
    values = df[value_col].astype(float)

    mean_val = float(values.mean())
    std_val = float(values.std())   # 标准差（论文常用）
    var_val = float(values.var())   # 方差

    if not use_div7:
        print(f"{mean_val:.4f} ± {std_val:.4f}")
    else:
        values_div7 = values / 7.0
        mean_div7 = float(values_div7.mean())
        std_div7 = float(values_div7.std())
        print(f"{mean_div7:.4f} ± {std_div7:.4f}")


if __name__ == "__main__":
    compute_value_stats(
        input_file="/home/coolas-fly/MARLlib/src/datas/Finish_time_Ours_without_relational_encoder_evaluation.csv",
        use_div7=False  # 改为 True 可输出除以7后的结果
    )