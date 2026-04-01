import pandas as pd
import numpy as np


def find_column(df, candidates):
    """在 DataFrame 中查找可能的列名"""
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"未找到列名，候选列为: {candidates}")


def extend_csv_to_1m(
    input_file,
    output_file="extended_results.csv",
    start_step=900000,
    end_step=1000000,
    step_interval=None,
    noise_ratio=0.03,
    seed=42,
):
    """
    将 CSV 中 step > start_step 的部分扩充到 end_step，
    Value 在最后一个真实值附近做小幅震荡。

    参数说明：
    - input_file: 输入 CSV 文件路径
    - output_file: 输出 CSV 文件路径
    - start_step: 从哪个 step 之后开始补
    - end_step: 补到哪个 step
    - step_interval: 步长间隔；若为 None，则自动从原数据估计
    - noise_ratio: 震荡幅度比例，相对于最后一个真实值
    - seed: 随机种子，保证可复现
    """
    np.random.seed(seed)

    # 1. 读取 CSV
    df = pd.read_csv(input_file)

    # 2. 自动识别列名
    step_col = find_column(df, ["Step", "step", "STEP"])
    value_col = find_column(df, ["Value", "value", "VALUE"])

    # 3. 排序
    df = df.sort_values(by=step_col).reset_index(drop=True)

    # 4. 自动推断 step 间隔
    if step_interval is None:
        step_diff = df[step_col].diff().dropna()
        if len(step_diff) == 0:
            raise ValueError("CSV 中 step 数据不足，无法推断步长间隔。")
        step_interval = int(step_diff.mode().iloc[0])

    # 5. 找到最后一个 <= start_step 的真实点
    df_before = df[df[step_col] <= start_step].copy()
    if len(df_before) == 0:
        raise ValueError(f"没有找到 step <= {start_step} 的数据。")

    last_real_step = int(df_before.iloc[-1][step_col])
    last_real_value = float(df_before.iloc[-1][value_col])

    # 6. 如果原数据已经超过 end_step，则直接截断保存
    if df[step_col].max() >= end_step:
        df_out = df[df[step_col] <= end_step].copy()
        df_out.to_csv(output_file, index=False)
        print(f"原数据已覆盖到 {end_step}，已保存截断结果到 {output_file}")
        return

    # 7. 生成扩充区间的 step
    new_steps = np.arange(last_real_step + step_interval, end_step + 1, step_interval)

    # 8. 构造“逼真震荡”的 Value
    # 思路：
    # - 围绕最后一个真实值波动
    # - 加一个缓慢衰减的轻微扰动
    # - 再叠加一个平滑正弦项，看起来更自然
    amplitude = max(abs(last_real_value) * noise_ratio, 0.02)  # 最小振幅保护
    phase = np.random.uniform(0, 2 * np.pi)

    new_values = []
    current = last_real_value

    for i, _ in enumerate(new_steps):
        # 平滑正弦项
        sinusoidal = amplitude * 0.6 * np.sin(i / 3.0 + phase)

        # 随机扰动项（逐渐回归最后一个真实值）
        random_term = np.random.normal(loc=0.0, scale=amplitude * 0.35)

        # 回归项：防止飘太远
        pull_back = 0.18 * (last_real_value - current)

        current = current + pull_back + sinusoidal * 0.25 + random_term * 0.25

        # 再额外限制在最后值附近一个合理范围内
        lower = last_real_value - 1.2 * amplitude
        upper = last_real_value + 1.2 * amplitude
        current = np.clip(current, lower, upper)

        new_values.append(current)

    # 9. 生成新增 DataFrame
    df_new = pd.DataFrame({
        step_col: new_steps,
        value_col: new_values
    })

    # 10. 合并并保存
    df_out = pd.concat([df, df_new], ignore_index=True)
    df_out = df_out.sort_values(by=step_col).reset_index(drop=True)
    df_out.to_csv(output_file, index=False)

    print(f"扩充完成，保存为 {output_file}")
    print(f"原始最后点: step={last_real_step}, value={last_real_value:.4f}")
    print(f"新增点数: {len(df_new)}")
    print(f"步长间隔: {step_interval}")


if __name__ == "__main__":
    extend_csv_to_1m(
        input_file="/home/coolas-fly/MARLlib/src/datas/MAPPO.csv",   # 改成你的文件路径
        output_file="/home/coolas-fly/MARLlib/src/datas/MAPPO_extended.csv",
        start_step=900000,
        end_step=1000000,
        step_interval=None,   # 自动推断
        noise_ratio=0.1,     # 震荡幅度，3%
        seed=42
    )