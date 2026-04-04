import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os


def build_alpha_curve(segment_defs):
    """Build a smooth, piecewise alpha(t) curve.

    segment_defs: list of tuples
        (length, start_alpha, end_alpha, amp)
    """
    parts = []
    for length, a0, a1, amp in segment_defs:
        x = np.linspace(0.0, 1.0, length)
        # 用 smoothstep 做更平滑的段间过渡
        smooth = 3.0 * x**2 - 2.0 * x**3
        base = a0 + (a1 - a0) * smooth
        # 更低频、更小幅的扰动，使曲线平滑且不完全规则
        wobble = amp * np.sin(1.6 * np.pi * x + 0.5) + 0.35 * amp * np.sin(3.2 * np.pi * x + 1.1)
        seg = np.clip(base + wobble, 0.0, 1.0)
        parts.append(seg)
    return np.concatenate(parts)


def plot_alpha_curves(alpha_list, labels=None, save_path="alpha_three_uavs.png", dpi=300):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    if labels is None:
        labels = [f"UAV {i+1}" for i in range(len(alpha_list))]

    for i, alpha in enumerate(alpha_list):
        t = np.arange(len(alpha))
        ax.plot(t, alpha, linewidth=1.8, color=colors[i], label=labels[i])

    ax.set_xlabel("Time Step")
    ax.set_ylabel(r"$\alpha(t)$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(r"Three-UAV $\alpha(t)$ Curves")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right", frameon=True)

    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    base, _ = os.path.splitext(save_path)
    plt.savefig(base + ".svg", bbox_inches="tight")
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.show()


def plot_uav_trajectories(
    traj_list,
    targets=None,
    regions=None,
    map_size=(100, 100),
    title="Three-UAV Trajectories",
    save_path="uav_trajectories.png",
    show_start_end=True,
    dpi=300,
):
    """
    画 3 架无人机的二维轨迹图

    参数说明：
    ----------
    traj_list : list
        长度为 3 的列表，每个元素是 shape=(T,2) 的 numpy 数组，
        例如 traj_list = [traj_uav1, traj_uav2, traj_uav3]

    targets : list[tuple], optional
        目标真实位置列表，如 [(x1, y1), (x2, y2), ...]

    regions : list[tuple], optional
        潜在目标区域列表，如 [(cx, cy, r), ...]
        这里默认用圆形区域表示

    map_size : tuple
        地图范围 (width, height)

    title : str
        图标题

    save_path : str
        输出图片路径

    show_start_end : bool
        是否显示起点和终点

    dpi : int
        图片分辨率
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # ===== 颜色和标签 =====
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["UAV 1", "UAV 2", "UAV 3"]

    # ===== 先画潜在目标区域（与 world_targets.py 一致：矩形区域） =====
    if regions is not None:
        for i, (xmin, xmax, ymin, ymax) in enumerate(regions):
            rect = Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.add_patch(rect)
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            ax.text(cx, ymax + 1.5, f"Region {i+1}", fontsize=10, ha="center")

    # ===== 画目标 =====
    if targets is not None:
        targets = np.array(targets)
        ax.scatter(
            targets[:, 0],
            targets[:, 1],
            marker="*",
            s=120,
            color="red",
            label="Targets",
            zorder=5
        )

    # ===== 画三条 UAV 轨迹（单一实线，不区分线型） =====
    for i, traj in enumerate(traj_list):
        traj = np.asarray(traj)

        ax.plot(
            traj[:, 0], traj[:, 1],
            linewidth=1.5,
            color=colors[i],
            label=labels[i]
        )

        if show_start_end:
            ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=80, color=colors[i], edgecolors="black")
            ax.scatter(traj[-1, 0], traj[-1, 1], marker="s", s=70, color=colors[i], edgecolors="black")

            ax.text(traj[0, 0], traj[0, 1] + 1.0, f"S{i+1}", fontsize=10, ha="center")
            ax.text(traj[-1, 0], traj[-1, 1] + 1.0, f"E{i+1}", fontsize=10, ha="center")

    # ===== 坐标轴与排版 =====
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="center right", bbox_to_anchor=(-0.08, 0.8), frameon=True)

    plt.tight_layout(rect=[0.15, 0, 1, 1])
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    # 保存为高分辨率位图
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # 同时保存为矢量图（论文推荐）
    base, _ = os.path.splitext(save_path)
    plt.savefig(base + ".svg", bbox_inches="tight")
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # ==========================================================
    # 下面这部分先放示例数据
    # 你后面只需要把 traj_uav1 / 2 / 3 替换成自己的轨迹即可
    # ==========================================================

    # ==========================================================
    # α(t) 曲线：与双分支门控含义一致的示意性时序
    # 高 α：更依赖 coarse branch（区域接近 / 全局引导）
    # 低 α：更依赖 fine branch（区域内精细搜索 / 目标附近）
    # ==========================================================
    alpha_uav1 = build_alpha_curve([
        (18, 0.88, 0.73, 0.020),   # approach Region 1
        (10, 0.57, 0.41, 0.016),   # sweep Region 1
        (15, 0.79, 0.65, 0.018),   # move to Region 3
        (11, 0.48, 0.33, 0.014),   # sweep Region 3
    ])

    alpha_uav2 = build_alpha_curve([
        (17, 0.82, 0.69, 0.018),   # approach Region 5
        (16, 0.54, 0.36, 0.016),   # sweep Region 5
        (12, 0.76, 0.62, 0.017),   # move to Region 6
        (15, 0.50, 0.31, 0.016),   # sweep Region 6
        (12, 0.73, 0.55, 0.017),   # move to Region 7
        (6, 0.24, 0.11, 0.010),    # tracking near target
    ])

    alpha_uav3 = build_alpha_curve([
        (12, 0.91, 0.78, 0.019),   # approach Region 2
        (13, 0.63, 0.46, 0.016),   # sweep Region 2
        (15, 0.84, 0.70, 0.018),   # move to Region 4
        (15, 0.56, 0.38, 0.015),   # sweep Region 4
    ])

    plot_alpha_curves(
        alpha_list=[alpha_uav1, alpha_uav2, alpha_uav3],
        labels=["UAV 1", "UAV 2", "UAV 3"],
        save_path="/home/coolas-fly/MARLlib/src/datas/figures/alpha_three_uavs.png",
        dpi=300,
    )

    traj_uav1 = np.array([
        # ===== Region 1 approach (强抖动接近) =====
        [8,8],[10,9],[9,11],[11,12],[10,14],[12,15],[14,16],[13,18],[15,19],[17,20],
        [16,21],[18,22],[20,21],[19,23],[21,24],[22,23],[21,25],[22,24],

        # ===== Region 1 sweep (斜向扫描，接近目标即结束) =====
        [17,16.5],[18,17],[19,17.5],
        [18.7,18.2],[17.9,18.9],[17.1,19.5],[16.3,20.0],
        [17.2,20.8],[18.0,21.4],[18.8,21.8],

        # ===== Move to Region 3（更隐式、更抖动） =====
        [21,24],[23,25],[24,27],[26,28],[27,30],[29,31],[28,33],[30,34],
        [32,35],[33,37],[35,38],[36,39],[38,40],[39,39.3],[40,38.6],

        # ===== Region 3 sweep（尽量向 frontier 扩展，不明显回头） =====
        [38,37.0],[39,37.5],
        [39.6,38.2],[38.8,37.0],[37.9,37.7],[37.0,40.3],
        [38.1,40.8],[39.2,41.0],[40.0,40.4],[40.8,39.6],[41.3,38.8]
    ])

    traj_uav2 = np.array([
        # ===== Approach Region 5（更隐式、更抖动） =====
        [90,10],[88,12],[89,14],[87,15],[85,17],[86,19],[84,20],[82,22],[80,23],[78,25],
        [76,24],[74,26],[72,28],[73,27],[72,29],[71,28],[70,30],

        # ===== Region 5 sweep（探索更充分，尽量不回头） =====
        [63,21.7],[63,24.3],[64,22.9],
        [65,23.6],[66,24.0],[67,24.6],[68,24.2],
        [67.2,26.0],[66.2,26.8],[65.1,27.5],[64.0,28.1],
        [62.9,28.6],[61.8,29.0],[62.8,29.4],

        # ===== Move to Region 6（不规则抖动，方向不固定） =====
        [62,31],[63.6,33.2],[65.8,34.1],[64.9,36.5],[67.3,37.4],[69.1,39.8],[68.0,41.0],[70.6,42.7],[72.2,44.1],[71.4,45.5],[73.8,46.7],[75.1,48.0],

        # ===== Region 6 sweep（探索更充分，尽量朝 frontier 大处扩展） =====
        [72,46],[73,47],[74,47.7],[75,48.4],[76,49.0],
        [77,49.6],[78,50.2],[79,50.8],[80,51.2],
        [79.3,52.0],[78.2,52.8],[77.0,53.6],[75.8,54.2],
        [74.6,54.8],[73.5,55.2],

        # ===== Move to Region 7（不规则抖动，方向不固定） =====
        [75,56],[76.7,57.6],[78.4,59.1],[77.6,61.4],[79.8,62.9],[81.1,65.6],[80.2,67.1],[82.5,69.8],[83.4,71.5],[82.6,73.8],[84.3,75.4],[84.0,78.0],

        # ===== Tracking =====
        [83,79],[84,80],[85,79.4],[85.4,78.4],[84.6,77.6],[83.8,78.2]
    ])

    traj_uav3 = np.array([
        # ===== Approach Region 2 (强抖动) =====
        [10,90],[12,88],[11,86],[13,84],[15,83],[14,81],[16,80],[18,79],[17,77],[19,76],[18,74],[16,74],

        # ===== Region 2 sweep（尽量前向扩展） =====
        [12,70],[13,71],[14,71.6],[15,72.2],[16,72.8],
        [17,73.5],[18,74.0],[18.6,74.8],[17.8,75.6],
        [16.8,76.2],[15.6,76.8],[14.4,77.3],[13.4,77.7],

        # ===== Move to Region 4（绕开 Region 3，从上侧进入，大小抖动混合） =====
        [17,77.8],[19.5,78.9],[18.2,77.4],[21.0,78.6],[23.8,77.1],[24.6,80.1],
        [25.7,77.8],[27.4,78.4],[30.2,79.6],[29.6,78.2],[31.8,79.0],[30.7,78.5],
        [34.1,79.4],[36.0,78.6],[35.1,79.8],[38.7,79.0],[40.5,79.4],[41.7,78.7],[44.0,78.0],

        # ===== Region 4 sweep（尽量前向扩展，不明显回头） =====
        [42,74],[43,75],[44,75.7],[45,76.3],[46,76.9],
        [47,77.4],[48,77.9],[49,78.3],
        [48.2,79.0],[47.0,79.5],[45.8,80.0],[44.6,80.3],[43.5,80.1],[44.2,79.2],[45.0,78.2]
    ])

    targets = [
        (18, 16),
        (14, 71),
        (42, 38),
        (45, 79),
        (68, 24),
        (80, 51),
        (84, 81),
    ]

    regions = [
        (14, 24, 14, 24),
        (10, 20, 68, 78),
        (34, 44, 34, 44),
        (42, 52, 72, 82),
        (60, 70, 20, 30),
        (72, 82, 46, 56),
        (78, 88, 74, 84),
    ]

    plot_uav_trajectories(
        traj_list=[traj_uav1, traj_uav2, traj_uav3],
        targets=targets,
        regions=regions,
        map_size=(100, 100),
        title="Three-UAV Collaborative Reconnaissance over Seven 1:1 Target Regions",
        save_path="/home/coolas-fly/MARLlib/src/datas/figures/three_uav_trajectories.png",
        show_start_end=True,
        dpi=300,
    )