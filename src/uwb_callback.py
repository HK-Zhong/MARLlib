# /home/coolas-fly/MARLlib/src/uwb_callback.py
#
# 目标：将环境在 info 中产出的 team-level 指标，按照 RLlib 官方推荐的方式
#      记录到 episode.custom_metrics（不使用 episode.hist_data，避免 result.json 里出现 hist_stats）。
#
# 兼容：Ray/RLlib 1.x 与 2.x 的 callbacks import 路径差异。

from __future__ import annotations

from typing import Any, Dict, Optional

# -----------------------------
# RLlib callbacks import (兼容)
# -----------------------------
try:
    # Ray 2.x (官方文档常用)
    from ray.rllib.algorithms.callbacks import DefaultCallbacks  # type: ignore
except Exception:  # pragma: no cover
    # Ray 1.x
    from ray.rllib.agents.callbacks import DefaultCallbacks  # type: ignore


# -----------------------------
# 你关心的 team 指标（10 个）
# -----------------------------
# 每个指标的聚合策略：
#  - "last" : 该 episode 内最后一次观测到的值（适合 ratio、最终状态）
#  - "sum"  : episode 内累加（适合“新增/计数”类）
#  - "max"  : episode 内取最大（适合 max-over-time 的强度类）
#  - "mean" : episode 内求均值（适合每步比例类）

METRIC_SPEC: Dict[str, str] = {
    "completion_ratio": "last",
    "team_new_targets_found": "sum",
    "episode_steps_to_done": "last",
    "first_target_time": "last",
    "last_target_time": "last",
    "role_balance": "last",
    "overlap_ratio": "mean",
    "collision_count_team": "sum",
    "new_cell_visits_team": "sum",
    "unsafe_ratio": "mean",
}


def _to_float(v: Any) -> Optional[float]:
    """尽量把 v 转成 float；失败返回 None（避免训练崩）。"""
    if v is None:
        return None
    try:
        # bool 也是 int 的子类，先转 int 再 float，避免 True/False 进来变成 1.0/0.0 也无所谓
        if isinstance(v, (int, bool)):
            return float(int(v))
        return float(v)
    except Exception:
        return None


def _extract_team_metrics(info: Any) -> Optional[Dict[str, Any]]:
    """从 env 的 info dict 中提取 team_metrics（兼容多种 wrapper 结构）。

    支持：
      - info["team_metrics"] == {...}
      - info["__all__"] == {...}  （如果你曾经这么塞过，但注意：env 返回给 RLlib 的 infos 的 key 不能包含 '__all__'
                                   这里仅做兼容读取，不建议 env 侧保留 '__all__'）
      - info 本身就是 {...}（直接当作 team metrics）
    """
    if not isinstance(info, dict):
        return None

    if isinstance(info.get("team_metrics"), dict):
        return info["team_metrics"]

    if isinstance(info.get("__all__"), dict):
        return info["__all__"]

    # 兜底：如果 info 里刚好就有这些 key，也可以直接当 team metrics
    hit = any(k in info for k in METRIC_SPEC.keys())
    if hit:
        return info

    return None


class UWBCustomMetricsCallbacks(DefaultCallbacks):
    """把 env.info -> episode.custom_metrics，写入 result.json。

    - 在 on_episode_step 中：
        1) 从任意 agent 的 last_info_for(agent_id) 中取 team_metrics。
        2) 根据 METRIC_SPEC 做在线累计（存到 episode.user_data）。

    - 在 on_episode_end 中：
        把累计结果写入 episode.custom_metrics。
        RLlib 会自动在 result.json 中输出：custom_metrics/<key>_{mean,min,max}
    """

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        # 初始化 user_data 存储
        episode.user_data.setdefault("uwb_sum", {})
        episode.user_data.setdefault("uwb_count", {})
        episode.user_data.setdefault("uwb_last", {})
        episode.user_data.setdefault("uwb_max", {})

    def on_episode_step(self, *, worker, base_env, episode, env_index=None, **kwargs):
        # 1) 找到一个 agent 的 info
        info = None
        try:
            for agent_id in episode.get_agents():
                info = episode.last_info_for(agent_id)
                if isinstance(info, dict) and info:
                    break
        except Exception:
            info = None

        team = _extract_team_metrics(info)
        if not isinstance(team, dict):
            return

        uwb_sum: Dict[str, float] = episode.user_data.get("uwb_sum", {})
        uwb_cnt: Dict[str, int] = episode.user_data.get("uwb_count", {})
        uwb_last: Dict[str, float] = episode.user_data.get("uwb_last", {})
        uwb_max: Dict[str, float] = episode.user_data.get("uwb_max", {})

        # 2) 在线累计
        for k, mode in METRIC_SPEC.items():
            if k not in team:
                continue
            v = _to_float(team.get(k))
            if v is None:
                continue

            # 聚合策略
            if mode == "last":
                uwb_last[k] = v
            elif mode == "sum":
                uwb_sum[k] = uwb_sum.get(k, 0.0) + v
            elif mode == "max":
                uwb_max[k] = v if (k not in uwb_max) else max(uwb_max[k], v)
            elif mode == "mean":
                uwb_sum[k] = uwb_sum.get(k, 0.0) + v
                uwb_cnt[k] = uwb_cnt.get(k, 0) + 1
            else:
                # 未知 mode，当 last 处理
                uwb_last[k] = v

        # 写回（episode.user_data 是 dict，可变对象；但写回更稳）
        episode.user_data["uwb_sum"] = uwb_sum
        episode.user_data["uwb_count"] = uwb_cnt
        episode.user_data["uwb_last"] = uwb_last
        episode.user_data["uwb_max"] = uwb_max

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        uwb_sum: Dict[str, float] = episode.user_data.get("uwb_sum", {})
        uwb_cnt: Dict[str, int] = episode.user_data.get("uwb_count", {})
        uwb_last: Dict[str, float] = episode.user_data.get("uwb_last", {})
        uwb_max: Dict[str, float] = episode.user_data.get("uwb_max", {})

        # 将最终聚合结果写入 custom_metrics
        for k, mode in METRIC_SPEC.items():
            if mode == "last":
                if k in uwb_last:
                    episode.custom_metrics[k] = uwb_last[k]
            elif mode == "sum":
                if k in uwb_sum:
                    episode.custom_metrics[k] = uwb_sum[k]
            elif mode == "max":
                if k in uwb_max:
                    episode.custom_metrics[k] = uwb_max[k]
            elif mode == "mean":
                if k in uwb_sum:
                    c = max(1, int(uwb_cnt.get(k, 0)))
                    episode.custom_metrics[k] = uwb_sum[k] / float(c)
            else:
                if k in uwb_last:
                    episode.custom_metrics[k] = uwb_last[k]

        # （可选）你也可以把 episode 长度写成 custom metric，方便 sanity check
        # episode.custom_metrics["episode_len"] = float(episode.length)