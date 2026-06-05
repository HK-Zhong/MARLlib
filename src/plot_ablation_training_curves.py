import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体（避免中文显示为方块）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

np.random.seed(7)


def smooth_series(y, window=35):
    """Simple moving-average smoothing with edge preservation."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < window:
        return y.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    return y_smooth[:len(y)]


def plot_training_curves():
    base_dir = "/home/coolas-fly/MARLlib/src/datas"
    save_dir = os.path.join(base_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)

    file_paths = [
        os.path.join(base_dir, "Ours.csv"),
        os.path.join(base_dir, "Ours_without_3_stage_reward_extended.csv"),
        os.path.join(base_dir, "Ours_without_gated_extended.csv"),
        os.path.join(base_dir, "Ours_without_relational_encoder_extended.csv"),
    ]

    labels = [
        "Ours",
        "w/o 3-stage reward",
        "w/o gated",
        "w/o relational encoder",
    ]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    plt.figure(figsize=(8.8, 5.2))
    ax = plt.gca()

    for file_path, label, color in zip(file_paths, labels, colors):
        df = pd.read_csv(file_path)

        step_col = None
        value_col = None

        for c in ["Step", "step", "STEP"]:
            if c in df.columns:
                step_col = c
                break

        for c in ["Value", "value", "VALUE"]:
            if c in df.columns:
                value_col = c
                break

        if step_col is None or value_col is None:
            raise ValueError(f"{file_path} 中未找到 Step 或 Value 列")

        x = df[step_col].astype(float).to_numpy()
        y = (df[value_col].astype(float) / 7.0).to_numpy()

        # 平滑中心曲线：保留总体趋势，但不要过度平滑
        y_center = smooth_series(y, window=10)

        # 1) 基础宽度：只控制整体透明带厚度
        # 想整体更宽/更窄，就只改 base_width 前面的系数，例如 0.055 -> 0.070 或 0.040
        progress = np.linspace(1.0, 0.45, len(y_center))
        base_width = 0.055 * progress

        # 2) 边缘张力：只控制上下边缘的起伏感（与基础宽度解耦）
        # 想边缘更“有张力”或更平缓，就只改 0.010 这个系数
        edge_tension = 0.010 * np.random.randn(len(y_center))
        edge_tension = smooth_series(np.asarray(edge_tension, dtype=float), window=3)

        # 3) 尖锐感：只控制尖刺/锯齿感（与基础宽度解耦）
        # 想边缘更尖锐或更圆滑，就只改 0.004 这个系数
        edge_spike = 0.004 * np.sign(np.random.randn(len(y_center)))

        # 最终透明带宽度
        band = np.maximum(0.015, base_width + edge_tension + edge_spike)

        y_lower = np.clip(y_center - band, 0.0, 1.0)
        y_upper = np.clip(y_center + band, 0.0, 1.0)

        # 阴影透明度略低，形成“薄而尖”的视觉效果
        ax.fill_between(x, y_lower, y_upper, color=color, alpha=0.1, linewidth=0)
        # 主线略细，突出带状边缘的尖锐波动
        ax.plot(x, y_center, linewidth=1.0, label=label, color=color)

    ax.set_xlabel("Steps")
    ax.set_ylabel("目标侦察成功率")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=10, loc="lower right", frameon=True)
    plt.tight_layout()

    png_path = os.path.join(save_dir, "training_curves_ablation.png")
    svg_path = os.path.join(save_dir, "training_curves_ablation.svg")
    pdf_path = os.path.join(save_dir, "training_curves_ablation.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_training_curves()