import numpy as np
from marllib.envs.base_env.my_env.world_base import UWBPlanningWorld
from marllib.envs.base_env.my_env.mpe_core import BaseScenario, Agent
from typing import Dict, Any


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self._ep_step = 0
        self._first_target_time = None
        self._last_target_time = None
        self._steps_to_done = None
        self._unsafe_steps = 0
        self._prev_target_found = None
        self._agent_target_counts = None
        self._prev_region_min_dist = None
        self._reward_target_total = 0.0
        self._reward_fine_search_total = 0.0
        self._reward_region_total = 0.0
        self._reward_explore_total = 0.0
        self._reward_safety_total = 0.0
        self._reward_collision_total = 0.0
        self._reward_repeat_total = 0.0

    def make_world(self, agent_num=3):
        world = UWBPlanningWorld(map_size=50.0, map_resolution=0.5)
        # set any world properties first
        num_agents = agent_num
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.2
            agent.max_speed = 3.0

        world.reset()
        self._reset_metrics(world)
        return world

    def reset_world(self, world, np_random):
        world.reset(np_random)
        self._reset_metrics(world)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def _reset_metrics(self, world):
        self._ep_step = 0
        self._first_target_time = None
        self._last_target_time = None
        self._steps_to_done = None
        self._unsafe_steps = 0

        # Track target completion deltas
        tf = getattr(world, "target_found", None)
        if tf is not None:
            self._prev_target_found = np.array(tf, dtype=bool)
        else:
            self._prev_target_found = None

        # Per-agent target attribution (for role_balance)
        n_agents = len(getattr(world, "agents", []))
        self._agent_target_counts = np.zeros((n_agents,), dtype=np.int32)
        self._prev_region_min_dist = np.full((n_agents,), np.nan, dtype=np.float32)

        # Reset reward accumulators for episode-level diagnostics
        self._reward_target_total = 0.0
        self._reward_fine_search_total = 0.0
        self._reward_region_total = 0.0
        self._reward_explore_total = 0.0
        self._reward_safety_total = 0.0
        self._reward_collision_total = 0.0
        self._reward_repeat_total = 0.0

        # Store metrics on world for logging/debug
        world.last_metrics = {}

        # Reset visible target progress tracking for all agents
        for a in getattr(world, "agents", []):
            a.prev_min_target_dist = None
            a.current_target_grid = None
            a.prev_visible_target_count = 0
            a.prev_frontier_strength = None
            if hasattr(a, "current_target_id"):
                a.current_target_id = None

    def _compute_local_frontier_strength(self, agent, world, patch_radius=7):
        """Compute local frontier strength around the agent.

        Frontier cells are defined as locally observed free cells that are
        adjacent to at least one unknown cell within the local window.
        """
        pgm = getattr(agent, "perceived_grid_map", None)
        if pgm is None:
            return 0.0

        ax, ay = agent.state.p_pos
        gx, gy = world.to_grid(float(ax), float(ay))

        R = int(patch_radius)
        H, W = pgm.shape
        x0, x1 = gx - R, gx + R + 1
        y0, y1 = gy - R, gy + R + 1

        sx0, sx1 = max(0, x0), min(H, x1)
        sy0, sy1 = max(0, y0), min(W, y1)

        patch = pgm[sx0:sx1, sy0:sy1]

        pad_left = sx0 - x0
        pad_right = x1 - sx1
        pad_bottom = sy0 - y0
        pad_top = y1 - sy1

        if pad_left or pad_right or pad_bottom or pad_top:
            patch = np.pad(
                patch,
                ((pad_left, pad_right), (pad_bottom, pad_top)),
                mode="constant",
                constant_values=1,
            )

        # In perceived_grid_map, free cells are 0 and unknown cells are 1.
        free_mask = (patch == 0)
        unknown_mask = (patch == 1)

        unknown_neighbor = np.zeros_like(unknown_mask, dtype=bool)
        unknown_neighbor[1:, :] |= unknown_mask[:-1, :]
        unknown_neighbor[:-1, :] |= unknown_mask[1:, :]
        unknown_neighbor[:, 1:] |= unknown_mask[:, :-1]
        unknown_neighbor[:, :-1] |= unknown_mask[:, 1:]

        frontier_mask = free_mask & unknown_neighbor
        return float(frontier_mask.sum())

    def _update_metrics_once_per_step(self, world):
        """Compute and store metrics ONCE per env step.

        This is called from global_reward(), which is assumed to be evaluated once
        per environment step in MARLlib.
        """
        self._ep_step += 1

        # --- completion ratio / remaining ---
        if hasattr(world, "get_completion_ratio"):
            completion_ratio = float(world.get_completion_ratio())
        else:
            # fallback
            tf = getattr(world, "target_found", [])
            completion_ratio = float(np.mean(np.array(tf, dtype=np.float32))) if tf else 0.0

        # 如果发现了超过 6 个目标，并且当前最后发现时间没有被记录，那么就更新最后发现时间，当做任务完成时间
        if completion_ratio >= 0.6 and self._last_target_time is None:
            self._last_target_time = int(self._ep_step)

        team_new_targets_found = int(getattr(world, "team_new_targets_found", 0))

        # --- first/last target time ---
        if team_new_targets_found > 0:
            if self._first_target_time is None:
                self._first_target_time = int(self._ep_step)

        # --- episode_steps_to_done ---
        all_found = bool(world.all_targets_found()) if hasattr(world, "all_targets_found") else False
        if all_found and self._steps_to_done is None:
            self._steps_to_done = int(self._ep_step)

        # --- new_cell_visits_team ---
        new_cell_visits_team = 0
        for a in getattr(world, "agents", []):
            new_cell_visits_team += int(getattr(a, "last_new_free_count", 0))

        # --- collision_count_team (pairwise) ---
        collision_count_team = 0
        agents = getattr(world, "agents", [])
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                if getattr(agents[i], "collide", False) and getattr(agents[j], "collide", False):
                    if self.is_collision(agents[i], agents[j]):
                        collision_count_team += 1

        # --- unsafe_ratio (EDT-based) ---
        unsafe_this_step = 0
        if hasattr(world, "is_safe"):
            for a in agents:
                try:
                    x_real, y_real = a.state.p_pos
                    if not world.is_safe(float(x_real), float(y_real)):
                        unsafe_this_step += 1
                except Exception:
                    continue
        if unsafe_this_step > 0:
            self._unsafe_steps += 1
        unsafe_ratio = float(self._unsafe_steps) / float(max(self._ep_step, 1))

        # --- role_balance (who found targets) ---
        # Attribute newly found targets to the nearest agent at discovery time.
        tf = getattr(world, "target_found", None)
        if tf is not None:
            cur_tf = np.array(tf, dtype=bool)
            if self._prev_target_found is not None and cur_tf.shape == self._prev_target_found.shape:
                newly_found_idx = np.nonzero(cur_tf & (~self._prev_target_found))[0]
                if newly_found_idx.size > 0:
                    # ensure target real coords are available
                    target_xy = None
                    if hasattr(world, "_target_points_real_array"):
                        target_xy = getattr(world, "_target_points_real_array")
                    elif hasattr(world, "target_points_grid"):
                        # compute on the fly (small N)
                        tgr = getattr(world, "target_points_grid")
                        target_xy = np.array([world.to_real(gx, gy) for gx, gy in tgr], dtype=np.float32)

                    if target_xy is not None and len(agents) > 0:
                        for tid in newly_found_idx.tolist():
                            tx, ty = float(target_xy[tid, 0]), float(target_xy[tid, 1])
                            # find nearest agent
                            best_k = 0
                            best_d = None
                            for k, a in enumerate(agents):
                                ax, ay = a.state.p_pos
                                dx = float(ax) - tx
                                dy = float(ay) - ty
                                d2 = dx * dx + dy * dy
                                if best_d is None or d2 < best_d:
                                    best_d = d2
                                    best_k = k
                            if self._agent_target_counts is not None and best_k < self._agent_target_counts.shape[0]:
                                self._agent_target_counts[best_k] += 1

                self._prev_target_found = cur_tf

        # role_balance as coefficient-of-variation (std/mean). Lower is more balanced.
        role_balance = 0.0
        if self._agent_target_counts is not None and self._agent_target_counts.size > 0:
            mean = float(np.mean(self._agent_target_counts))
            std = float(np.std(self._agent_target_counts))
            role_balance = float(std / max(mean, 1e-6))

        # --- overlap_ratio (redundant exploration) ---
        # Use visited_grid_map if available: overlap = (sum_visited - unique_visited) / sum_visited
        sum_visited = 0
        unique_visited = 0
        visited_union = None
        for a in agents:
            v = getattr(a, "visited_grid_map", None)
            if v is None:
                continue
            v_bin = (v.astype(np.int8) > 0)
            sum_visited += int(v_bin.sum())
            if visited_union is None:
                visited_union = v_bin.copy()
            else:
                visited_union |= v_bin
        if visited_union is not None:
            unique_visited = int(visited_union.sum())
        overlap_ratio = 0.0
        if sum_visited > 0:
            overlap_ratio = float((sum_visited - unique_visited) / float(sum_visited))

        # --- store all metrics on world ---
        world.last_metrics = {
            "completion_ratio": float(completion_ratio),
            "team_new_targets_found": int(team_new_targets_found),
            "episode_step": int(self._ep_step),
            "episode_steps_to_done": int(self._steps_to_done) if self._steps_to_done is not None else -1,
            "first_target_time": int(self._first_target_time) if self._first_target_time is not None else -1,
            "last_target_time": int(self._last_target_time) if self._last_target_time is not None else -1,
            "role_balance": float(role_balance),
            "overlap_ratio": float(overlap_ratio),
            "collision_count_team": int(collision_count_team),
            "new_cell_visits_team": int(new_cell_visits_team),
            "unsafe_ratio": float(unsafe_ratio),
            "reward_target_total": float(self._reward_target_total),
            "reward_fine_search_total": float(self._reward_fine_search_total),
            "reward_region_total": float(self._reward_region_total),
            "reward_explore_total": float(self._reward_explore_total),
            "reward_safety_total": float(self._reward_safety_total),
            "reward_collision_total": float(self._reward_collision_total),
            "reward_repeat_total": float(self._reward_repeat_total),
        }

    def reward(self, agent, world):
        """
        Three-stage local reward design.

        Stage A: Coarse search (outside target region, target not visible)
            - use region_progress to guide the agent toward the nearest unfinished region
            - use reward_repeat to suppress useless wandering when the agent neither
              discovers new cells nor makes region progress

        Stage B: Fine search (inside target region, target still not visible)
            - disable region_progress shaping
            - disable shared exploration shaping
            - use reward_fine_search to encourage expanding local frontier inside the region

        Stage C: Target pursuit (target becomes visible)
            - disable region_progress shaping
            - give a one-time discovery bonus when target first becomes visible
            - use reward_visible_target to encourage moving closer to the visible target

        Shared penalties / shaping across stages:
            - exploration/info gain reward
            - collision penalty
            - safety penalty
            - per-step penalty
        """

        # =============================================================
        # 0) Shared low-level signals used across all reward stages
        # =============================================================

        # Number of currently visible anchors (currently disabled in reward via W_VISIBLE=0)
        visible_ids = getattr(agent, "last_visible_uwb_ids", [])
        num_visible = len(visible_ids)
        total_anchors = max(len(getattr(world, "uwb_locations", {})), 1)

        # Incremental exploration signal: how many new free cells were discovered this step.
        new_free = int(getattr(agent, "last_new_free_count", 0))
        denom = float(max(world.grid_size * world.grid_size, 1))
        info_gain = (float(new_free) / denom) if denom > 0 else 0.0

        # Pairwise collision count for local collision penalty.
        collisions = 0
        if getattr(agent, "collide", False):
            for other_agent in world.agents:
                if other_agent is agent:
                    continue
                if self.is_collision(other_agent, agent):
                    collisions += 1

        # Binary safety violation flag (e.g. too close to obstacles / unsafe area).
        unsafe = 0
        if hasattr(world, "is_safe"):
            x_real, y_real = agent.state.p_pos
            if not world.is_safe(float(x_real), float(y_real)):
                unsafe = 1

        # =============================================================
        # 1) Stage classifier: determine whether the agent is still in
        #    coarse search, has entered the target region, or already sees
        #    a concrete target.
        # =============================================================

        in_target_region = False

        # Dense region-progress shaping used ONLY during coarse search.
        # It measures whether the agent moves closer to the nearest unfinished
        # region center. Once inside the region (or once a target is visible),
        # this shaping will be disabled later.
        region_progress = 0.0
        cur_min_dist = np.nan

        if hasattr(world, "_region_centers") and hasattr(world, "target_found"):
            ax, ay = agent.state.p_pos
            found_mask = np.array(world.target_found, dtype=bool)

            if world._region_centers.shape[0] == found_mask.shape[0]:
                centers = world._region_centers[~found_mask]
            else:
                centers = world._region_centers

            if centers.size > 0:
                dx = centers[:, 0] - float(ax)
                dy = centers[:, 1] - float(ay)
                dist_sq = dx * dx + dy * dy
                cur_min_dist = float(np.sqrt(dist_sq.min()))

                # Region entry is determined by distance to the nearest unfinished
                # region center. This provides the switch from coarse search to
                # fine search.
                REGION_SEARCH_RADIUS_M = 5.0
                in_target_region = (cur_min_dist <= REGION_SEARCH_RADIUS_M)

                try:
                    agent_idx = int(agent.name.split("_")[-1])
                except Exception:
                    agent_idx = 0

                if (
                        self._prev_region_min_dist is not None
                        and 0 <= agent_idx < len(self._prev_region_min_dist)
                        and np.isfinite(self._prev_region_min_dist[agent_idx])
                ):
                    prev_dist = float(self._prev_region_min_dist[agent_idx])
                    # Positive when moving closer, negative when moving away.
                    region_progress = prev_dist - cur_min_dist

                if self._prev_region_min_dist is not None and 0 <= agent_idx < len(self._prev_region_min_dist):
                    self._prev_region_min_dist[agent_idx] = cur_min_dist

        # =============================================================
        # 2) Visible-target analysis: determine whether Stage C has started.
        #    Also compute the one-time discovery bonus and dense visible-target
        #    progress shaping with locked-target anti-jitter logic.
        # =============================================================

        has_visible_target = False
        reward_target_discovery = 0.0

        if not hasattr(agent, "prev_visible_target_count"):
            agent.prev_visible_target_count = 0

        # Locked visible-target progress reward:
        # - lock by stable grid coordinate instead of transient visible-list index
        # - if the locked target disappears, reset tracking
        # - once visible, reward moving closer to that target
        reward_visible_target = 0.0

        if not hasattr(agent, "current_target_grid"):
            agent.current_target_grid = None

        if hasattr(agent, "perceived_target_map"):
            visible_targets = np.argwhere(agent.perceived_target_map == 1)
            cur_visible_count = int(len(visible_targets))
            has_visible_target = cur_visible_count > 0

            newly_visible = max(
                cur_visible_count - int(getattr(agent, "prev_visible_target_count", 0)),
                0,
            )
            if newly_visible > 0:
                reward_target_discovery += float(newly_visible)

            if len(visible_targets) > 0:
                ax, ay = agent.state.p_pos

                # Convert visible target list to a set of stable grid-coordinate keys.
                visible_keys = {(int(gx_t), int(gy_t)) for gx_t, gy_t in visible_targets}

                # If the previously locked target is no longer visible, drop it.
                if agent.current_target_grid is not None:
                    cur_key = (int(agent.current_target_grid[0]), int(agent.current_target_grid[1]))
                    if cur_key not in visible_keys:
                        agent.current_target_grid = None
                        agent.prev_min_target_dist = None

                # Acquire a target only when unlocked.
                if agent.current_target_grid is None:
                    best_key = None
                    best_d2 = None
                    for gx_t, gy_t in visible_targets:
                        tx, ty = world.to_real(int(gx_t), int(gy_t))
                        dx = tx - ax
                        dy = ty - ay
                        d2 = dx * dx + dy * dy
                        if best_d2 is None or d2 < best_d2:
                            best_d2 = d2
                            best_key = (int(gx_t), int(gy_t))
                    agent.current_target_grid = best_key
                    agent.prev_min_target_dist = None

                # Compute reward only if a locked visible target exists.
                if agent.current_target_grid is not None:
                    gx_t, gy_t = int(agent.current_target_grid[0]), int(agent.current_target_grid[1])
                    tx, ty = world.to_real(gx_t, gy_t)

                    dx = tx - ax
                    dy = ty - ay
                    dist = np.sqrt(dx * dx + dy * dy)

                    if getattr(agent, "prev_min_target_dist", None) is None:
                        agent.prev_min_target_dist = dist
                    else:
                        delta = agent.prev_min_target_dist - dist
                        delta = np.clip(delta, -1.0, 1.0)
                        reward_visible_target += 2.0 * delta
                        agent.prev_min_target_dist = dist
            else:
                # No visible target -> fully reset tracking.
                agent.prev_min_target_dist = None
                agent.current_target_grid = None
                agent.prev_visible_target_count = 0

            agent.prev_visible_target_count = cur_visible_count if has_visible_target else 0

        # =============================================================
        # 3) Reward weights
        # =============================================================

        W_INFO = 0.1
        W_VISIBLE = 0.0
        W_COLLIDE = 1.0
        W_UNSAFE = 0.5
        W_STEP = 0.02
        W_REGION = 0.5
        W_TARGET_DISCOVERY = 1.0
        W_FINE_SEARCH = 3.0
        W_REPEAT_MARGIN = 0.05

        # =============================================================
        # 4) Stage-dependent reward construction
        #
        # Stage A: coarse search
        #   not in_target_region and not has_visible_target
        #   -> region guidance is active
        #
        # Stage B: fine search
        #   in_target_region and not has_visible_target
        #   -> region guidance is disabled
        #   -> shared exploration shaping is disabled
        #   -> reward_fine_search encourages expanding local frontier inside region
        #
        # Stage C: target pursuit
        #   has_visible_target
        #   -> region guidance is disabled
        #   -> reward_target = discovery bonus + visible-target progress reward
        #
        # NOTE:
        # - True team success reward is handled only in global_reward().
        # - This local reward provides stage-specific shaping and penalties.
        # =============================================================

        # -------------------------------------------------------------
        # 4.1 Shared reward terms
        # -------------------------------------------------------------

        # Disable region-center shaping once the agent has entered the region
        # (Stage B) or once a concrete target is visible (Stage C).
        if has_visible_target or in_target_region:
            region_progress = 0.0

        reward_explore = W_INFO * info_gain + W_VISIBLE * (num_visible / float(total_anchors)) - W_STEP

        # In Stage B, disable the shared exploration term and let frontier reward
        # dominate local fine-search behavior.
        if in_target_region and (not has_visible_target):
            reward_explore = 0.0

        reward_region = W_REGION * float(region_progress)
        reward_collision = -W_COLLIDE * float(collisions)
        reward_safety = -W_UNSAFE * float(unsafe)
        reward_target = reward_visible_target + W_TARGET_DISCOVERY * reward_target_discovery

        # -------------------------------------------------------------
        # 4.2 Stage-B specific reward: fine search inside target region
        # -------------------------------------------------------------
        reward_fine_search = 0.0
        if in_target_region and (not has_visible_target):
            frontier_strength = self._compute_local_frontier_strength(agent, world, patch_radius=7)
            prev_frontier = getattr(agent, "prev_frontier_strength", None)
            if prev_frontier is None:
                reward_fine_search = 0.0
            else:
                frontier_delta = frontier_strength - float(prev_frontier)
                reward_fine_search = W_FINE_SEARCH * float(frontier_delta)
            agent.prev_frontier_strength = float(frontier_strength)
        else:
            agent.prev_frontier_strength = None

        # -------------------------------------------------------------
        # 4.3 Stage-A specific reward: repeated-exploration penalty
        # -------------------------------------------------------------
        reward_repeat = 0.0
        if (not has_visible_target) and (not in_target_region) and (new_free == 0) and (region_progress <= 0.0):
            reward_repeat = -(abs(reward_region) + W_REPEAT_MARGIN)

        # =============================================================
        # 5) Final local reward sum
        # =============================================================

        rew = 0.0
        rew += reward_explore
        rew += reward_region
        rew += reward_collision
        rew += reward_safety
        rew += reward_target
        rew += reward_fine_search
        rew += reward_repeat

        # Episode-level diagnostic accumulators (sum over all per-agent reward calls)
        self._reward_explore_total += float(reward_explore)
        self._reward_region_total += float(reward_region)
        self._reward_collision_total += float(reward_collision)
        self._reward_safety_total += float(reward_safety)
        self._reward_target_total += float(reward_target)
        self._reward_fine_search_total += float(reward_fine_search)
        self._reward_repeat_total += float(reward_repeat)

        return float(rew)

    # def reward(self, agent, world):
    #     """
    #     Unified reward without explicit three-stage mechanism.
    #
    #     This version uses a single task-level reward throughout the whole episode.
    #     It keeps only the essential components:
    #       - exploration / information gain
    #       - visible target progress
    #       - target discovery bonus
    #       - collision penalty
    #       - safety penalty
    #       - per-step penalty
    #
    #     Removed:
    #       - region guidance reward
    #       - frontier-based fine-search reward
    #       - repeated-exploration stage-specific penalty
    #       - any explicit stage classification
    #     """
    #
    #     # =============================================================
    #     # 1) Shared low-level signals
    #     # =============================================================
    #
    #     # Visible anchors (currently optional, W_VISIBLE can be 0)
    #     visible_ids = getattr(agent, "last_visible_uwb_ids", [])
    #     num_visible = len(visible_ids)
    #     total_anchors = max(len(getattr(world, "uwb_locations", {})), 1)
    #
    #     # Exploration / information gain
    #     new_free = int(getattr(agent, "last_new_free_count", 0))
    #     denom = float(max(world.grid_size * world.grid_size, 1))
    #     info_gain = (float(new_free) / denom) if denom > 0 else 0.0
    #
    #     # Collision penalty
    #     collisions = 0
    #     if getattr(agent, "collide", False):
    #         for other_agent in world.agents:
    #             if other_agent is agent:
    #                 continue
    #             if self.is_collision(other_agent, agent):
    #                 collisions += 1
    #
    #     # Safety penalty
    #     unsafe = 0
    #     if hasattr(world, "is_safe"):
    #         x_real, y_real = agent.state.p_pos
    #         if not world.is_safe(float(x_real), float(y_real)):
    #             unsafe = 1
    #
    #     # =============================================================
    #     # 2) Target visibility and progress
    #     # =============================================================
    #
    #     reward_target_discovery = 0.0
    #     reward_visible_target = 0.0
    #     has_visible_target = False
    #
    #     if not hasattr(agent, "prev_visible_target_count"):
    #         agent.prev_visible_target_count = 0
    #     if not hasattr(agent, "current_target_grid"):
    #         agent.current_target_grid = None
    #
    #     if hasattr(agent, "perceived_target_map"):
    #         visible_targets = np.argwhere(agent.perceived_target_map == 1)
    #         cur_visible_count = int(len(visible_targets))
    #         has_visible_target = cur_visible_count > 0
    #
    #         # one-time discovery bonus
    #         newly_visible = max(
    #             cur_visible_count - int(getattr(agent, "prev_visible_target_count", 0)),
    #             0,
    #         )
    #         if newly_visible > 0:
    #             reward_target_discovery += float(newly_visible)
    #
    #         # target progress shaping
    #         if len(visible_targets) > 0:
    #             ax, ay = agent.state.p_pos
    #             visible_keys = {(int(gx_t), int(gy_t)) for gx_t, gy_t in visible_targets}
    #
    #             # reset locked target if disappeared
    #             if agent.current_target_grid is not None:
    #                 cur_key = (int(agent.current_target_grid[0]), int(agent.current_target_grid[1]))
    #                 if cur_key not in visible_keys:
    #                     agent.current_target_grid = None
    #                     agent.prev_min_target_dist = None
    #
    #             # lock nearest visible target
    #             if agent.current_target_grid is None:
    #                 best_key = None
    #                 best_d2 = None
    #                 for gx_t, gy_t in visible_targets:
    #                     tx, ty = world.to_real(int(gx_t), int(gy_t))
    #                     dx = tx - ax
    #                     dy = ty - ay
    #                     d2 = dx * dx + dy * dy
    #                     if best_d2 is None or d2 < best_d2:
    #                         best_d2 = d2
    #                         best_key = (int(gx_t), int(gy_t))
    #                 agent.current_target_grid = best_key
    #                 agent.prev_min_target_dist = None
    #
    #             # dense reward: moving closer to currently visible target
    #             if agent.current_target_grid is not None:
    #                 gx_t, gy_t = int(agent.current_target_grid[0]), int(agent.current_target_grid[1])
    #                 tx, ty = world.to_real(gx_t, gy_t)
    #                 dx = tx - ax
    #                 dy = ty - ay
    #                 dist = np.sqrt(dx * dx + dy * dy)
    #
    #                 if getattr(agent, "prev_min_target_dist", None) is None:
    #                     agent.prev_min_target_dist = dist
    #                 else:
    #                     delta = agent.prev_min_target_dist - dist
    #                     delta = np.clip(delta, -1.0, 1.0)
    #                     reward_visible_target += 2.0 * delta
    #                     agent.prev_min_target_dist = dist
    #         else:
    #             agent.prev_min_target_dist = None
    #             agent.current_target_grid = None
    #             agent.prev_visible_target_count = 0
    #
    #         agent.prev_visible_target_count = cur_visible_count if has_visible_target else 0
    #
    #     # no frontier logic in unified reward
    #     agent.prev_frontier_strength = None
    #
    #     # =============================================================
    #     # 3) Reward weights
    #     # =============================================================
    #
    #     W_INFO = 0.1
    #     W_VISIBLE = 0.0
    #     W_COLLIDE = 1.0
    #     W_UNSAFE = 0.5
    #     W_STEP = 0.02
    #     W_TARGET_DISCOVERY = 1.0
    #
    #     # =============================================================
    #     # 4) Unified reward construction
    #     # =============================================================
    #
    #     reward_explore = W_INFO * info_gain + W_VISIBLE * (num_visible / float(total_anchors)) - W_STEP
    #     reward_collision = -W_COLLIDE * float(collisions)
    #     reward_safety = -W_UNSAFE * float(unsafe)
    #     reward_target = reward_visible_target + W_TARGET_DISCOVERY * reward_target_discovery
    #
    #     rew = 0.0
    #     rew += reward_explore
    #     rew += reward_collision
    #     rew += reward_safety
    #     rew += reward_target
    #
    #     # =============================================================
    #     # 5) Logging accumulators
    #     # =============================================================
    #
    #     self._reward_explore_total += float(reward_explore)
    #     self._reward_region_total += 0.0
    #     self._reward_collision_total += float(reward_collision)
    #     self._reward_safety_total += float(reward_safety)
    #     self._reward_target_total += float(reward_target)
    #     self._reward_fine_search_total += 0.0
    #     self._reward_repeat_total += 0.0
    #
    #     return float(rew)

    def global_reward(self, world):
        """Simplified cooperative global reward (team-level).

        Keeps ONLY signals that truly reflect cooperation:
        + team-level newly found hidden targets (shared success)
        + strong completion bonus when all targets are found
        - inter-agent collision penalty (coordination)

        Note: Exploration shaping, visible-anchor shaping, safety penalty, step penalty,
        and region proximity shaping are handled in per-agent `reward()`.
        """
        # -------------------------
        # Team target discovery / completion
        # -------------------------

        team_new_targets = int(getattr(world, "team_new_targets_found", 0))
        all_found = bool(world.all_targets_found()) if hasattr(world, "all_targets_found") else False

        # -------------------------
        # Cooperative weights (global-only target success reward)
        # -------------------------
        W_TARGET = 20.0
        W_DONE_BONUS = 200.0

        rew = 0.0
        rew += W_TARGET * float(team_new_targets)
        if all_found:
            rew += W_DONE_BONUS

        # Count global cooperative reward into target-related diagnostics
        self._reward_target_total += float(rew)

        # Update per-step metrics once after all reward components are accumulated
        self._update_metrics_once_per_step(world)

        return float(rew)

    def global_state(self, world):
        """
        Build a richer centralized global state for critic use.

        The information content is unchanged, but the final flattened layout is
        reorganized into a more structured form for a future relational critic:

        1) agent_tokens_flat
           for each agent:
             [pos_x, pos_y, vel_x, vel_y, dist_to_nearest_unfinished_target]

        2) target_tokens_flat
           for each target:
             [target_x, target_y, found_flag, visible_flag, dist_to_nearest_agent]

        3) global_aux_flat
           [visited_low, grid_map_low]

        Returned shape:
            [global_state_dim, ]
        """

        # -------------------------------------------------
        # 1) All agents' normalized position and velocity
        # -------------------------------------------------
        half_map = float(world.map_size_m) / 2.0
        agent_kinematics = []
        for a in getattr(world, "agents", []):
            pos = a.state.p_pos.astype(np.float32) / max(half_map, 1e-6)
            vel = a.state.p_vel.astype(np.float32) / max(float(a.max_speed), 1e-6)
            agent_kinematics.append(np.concatenate([pos, vel], axis=0))

        if len(agent_kinematics) > 0:
            all_agent_pos_vel = np.concatenate(agent_kinematics, axis=0).astype(np.float32, copy=False)
        else:
            all_agent_pos_vel = np.zeros((0,), dtype=np.float32)

        # -------------------------------------------------
        # 2) All targets' normalized real positions (critic-only full state)
        # -------------------------------------------------
        tpr = getattr(world, "target_points_real", None)
        if tpr is None or len(tpr) == 0:
            target_pos_vec = np.zeros((0,), dtype=np.float32)
        else:
            target_pos = np.asarray(tpr, dtype=np.float32).reshape(-1, 2)
            target_pos_norm = target_pos / max(half_map, 1e-6)
            target_pos_vec = target_pos_norm.ravel().astype(np.float32, copy=False)

        # -------------------------------------------------
        # 3) Ground-truth target completion vector
        # -------------------------------------------------
        tf = getattr(world, "target_found", None)
        if tf is None:
            target_found_vec = np.zeros((0,), dtype=np.float32)
        else:
            target_found_vec = np.asarray(tf, dtype=np.float32).ravel()

        # -------------------------------------------------
        # 4) Each target's distance to the nearest agent
        # -------------------------------------------------
        if len(getattr(world, "agents", [])) > 0 and tpr is not None and len(tpr) > 0:
            agent_xy = np.array([a.state.p_pos for a in world.agents], dtype=np.float32).reshape(-1, 2)
            target_xy = np.asarray(tpr, dtype=np.float32).reshape(-1, 2)
            diff = target_xy[:, None, :] - agent_xy[None, :, :]  # [T, A, 2]
            dists = np.linalg.norm(diff, axis=-1)  # [T, A]
            nearest_agent_dist = dists.min(axis=1).astype(np.float32)  # [T]
            nearest_agent_dist = nearest_agent_dist / max(float(world.map_size_m), 1e-6)
        else:
            nearest_agent_dist = np.zeros((0,), dtype=np.float32)

        # -------------------------------------------------
        # 5) Low-resolution global visited map (team coverage summary)
        #    Use union of all agents' visited_grid_map, then average-pool to 15x15.
        # -------------------------------------------------
        visited_union = None
        for a in getattr(world, "agents", []):
            v = getattr(a, "visited_grid_map", None)
            if v is None:
                continue
            v_bin = (np.asarray(v) > 0).astype(np.float32)
            if visited_union is None:
                visited_union = v_bin.copy()
            else:
                visited_union = np.maximum(visited_union, v_bin)

        low_h, low_w = 15, 15
        if visited_union is None:
            visited_low = np.zeros((low_h * low_w,), dtype=np.float32)
        else:
            gh, gw = visited_union.shape
            block_h = max(gh // low_h, 1)
            block_w = max(gw // low_w, 1)
            trimmed = visited_union[:block_h * low_h, :block_w * low_w]
            pooled = trimmed.reshape(low_h, block_h, low_w, block_w).mean(axis=(1, 3))
            visited_low = pooled.astype(np.float32, copy=False).ravel()

        # -------------------------------------------------
        # 5.5) Low-resolution ground-truth obstacle map for critic use
        #      Downsample world.grid_map to the same 15x15 resolution.
        # -------------------------------------------------
        grid_map = getattr(world, "grid_map", None)
        if grid_map is None:
            grid_map_low = np.zeros((low_h * low_w,), dtype=np.float32)
        else:
            grid_bin = (np.asarray(grid_map) > 0).astype(np.float32)
            gh, gw = grid_bin.shape
            block_h = max(gh // low_h, 1)
            block_w = max(gw // low_w, 1)
            trimmed = grid_bin[:block_h * low_h, :block_w * low_w]
            pooled = trimmed.reshape(low_h, block_h, low_w, block_w).mean(axis=(1, 3))
            grid_map_low = pooled.astype(np.float32, copy=False).ravel()

        # -------------------------------------------------
        # 6) Each target's current visibility flag (visible to any agent)
        # -------------------------------------------------
        if tpr is not None and len(tpr) > 0 and len(getattr(world, "agents", [])) > 0:
            target_visible_any = np.zeros((len(tpr),), dtype=np.float32)
            for a in world.agents:
                ptm = getattr(a, "perceived_target_map", None)
                if ptm is None:
                    continue
                for tid, (gx_t, gy_t) in enumerate(getattr(world, "target_points_grid", [])):
                    if 0 <= int(gx_t) < ptm.shape[0] and 0 <= int(gy_t) < ptm.shape[1]:
                        if ptm[int(gx_t), int(gy_t)] > 0:
                            target_visible_any[tid] = 1.0
        else:
            target_visible_any = np.zeros((0,), dtype=np.float32)

        # -------------------------------------------------
        # 7) Each agent's distance to the nearest unfinished target
        # -------------------------------------------------
        if len(getattr(world, "agents", [])) > 0 and tpr is not None and len(tpr) > 0:
            found_mask = np.asarray(tf, dtype=bool).ravel() if tf is not None else np.zeros((len(tpr),), dtype=bool)
            target_xy = np.asarray(tpr, dtype=np.float32).reshape(-1, 2)
            unfinished_xy = target_xy[~found_mask] if found_mask.shape[0] == target_xy.shape[0] else target_xy

            if unfinished_xy.shape[0] > 0:
                agent_xy = np.array([a.state.p_pos for a in world.agents], dtype=np.float32).reshape(-1, 2)
                diff = agent_xy[:, None, :] - unfinished_xy[None, :, :]  # [A, T_unfinished, 2]
                dists = np.linalg.norm(diff, axis=-1)  # [A, T_unfinished]
                agent_to_nearest_unfinished = dists.min(axis=1).astype(np.float32)
                agent_to_nearest_unfinished = agent_to_nearest_unfinished / max(float(world.map_size_m), 1e-6)
            else:
                agent_to_nearest_unfinished = np.zeros((len(world.agents),), dtype=np.float32)
        else:
            agent_to_nearest_unfinished = np.zeros((0,), dtype=np.float32)

        # -------------------------------------------------
        # Structured packing for relational critic:
        #   1) agent_tokens_flat
        #   2) target_tokens_flat
        #   3) global_aux_flat 450
        # -------------------------------------------------
        n_agents = len(getattr(world, "agents", []))
        n_targets = len(tpr) if (tpr is not None) else 0

        # agent_tokens: [A, 5] = [pos_x, pos_y, vel_x, vel_y, dist_to_nearest_unfinished_target]
        if n_agents > 0 and all_agent_pos_vel.size == n_agents * 4 and agent_to_nearest_unfinished.size == n_agents:
            agent_pos_vel = all_agent_pos_vel.reshape(n_agents, 4)
            agent_dist = agent_to_nearest_unfinished.reshape(n_agents, 1)
            agent_tokens = np.concatenate([agent_pos_vel, agent_dist], axis=1).astype(np.float32, copy=False)
        else:
            agent_tokens = np.zeros((0, 5), dtype=np.float32)
        agent_tokens_flat = agent_tokens.ravel().astype(np.float32, copy=False)

        # target_tokens: [T, 5] = [x, y, found_flag, visible_flag, dist_to_nearest_agent]
        if (
                n_targets > 0
                and target_pos_vec.size == n_targets * 2
                and target_found_vec.size == n_targets
                and target_visible_any.size == n_targets
                and nearest_agent_dist.size == n_targets
        ):
            target_pos = target_pos_vec.reshape(n_targets, 2)
            target_found = target_found_vec.reshape(n_targets, 1)
            target_visible = target_visible_any.reshape(n_targets, 1)
            target_dist = nearest_agent_dist.reshape(n_targets, 1)
            target_tokens = np.concatenate(
                [target_pos, target_found, target_visible, target_dist], axis=1
            ).astype(np.float32, copy=False)
        else:
            target_tokens = np.zeros((0, 5), dtype=np.float32)
        target_tokens_flat = target_tokens.ravel().astype(np.float32, copy=False)

        # global auxiliary vector
        global_aux_flat = np.concatenate(
            [
                visited_low.astype(np.float32, copy=False).ravel(),
                grid_map_low.astype(np.float32, copy=False).ravel(),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        state_parts = [agent_tokens_flat, target_tokens_flat, global_aux_flat]
        state_debug_parts = [
            ("agent_tokens_flat", tuple(agent_tokens_flat.shape)),
            ("target_tokens_flat", tuple(target_tokens_flat.shape)),
            ("global_aux_flat", tuple(global_aux_flat.shape)),
        ]

        global_state = np.concatenate(state_parts, axis=0).astype(np.float32, copy=False)

        if not hasattr(self, "_global_state_debug_printed"):
            print("\n[GLOBAL STATE DEBUG] components:")
            for name, shape in state_debug_parts:
                print(f"  - {name}: {shape}")
            print("[GLOBAL STATE DEBUG] total global_state shape:", global_state.shape)
            self._global_state_debug_printed = True

        return global_state

    def observation(self, agent, world):
        """
        Sample observation design (normalized + fixed patch).

        Observation = [
            normalized agent position (2),          in [-1, 1] roughly
            normalized agent velocity (2),          scaled by a constant(max velocity)
            flattened local perceived grid map patch ( (2R+1) x (2R+1) )
        ]
        """

        # -------------------------------------------------
        # 1) Update visible target map (targets become visible when
        #      they enter the agent perception range)
        # -------------------------------------------------
        if hasattr(world, "update_target_visibility_for_agent"):
            world.update_target_visibility_for_agent(agent)

        # -------------------------------------------------
        # 2) Normalize agent kinematics (real-world coordinates)
        # -------------------------------------------------
        # position normalization: map is centered at (0,0) with side length map_size_m
        half_map = float(world.map_size_m) / 2.0
        pos = agent.state.p_pos.astype(np.float32)
        pos_norm = pos / max(half_map, 1e-6)

        # velocity normalization: use a conservative constant scale (m/s)
        vel = agent.state.p_vel.astype(np.float32)
        vel_norm = vel / agent.max_speed

        obs_parts = [pos_norm, vel_norm]
        obs_debug_parts = [
            ("pos_norm", tuple(pos_norm.shape)),
            ("vel_norm", tuple(vel_norm.shape)),
        ]

        # -------------------------------------------------
        # 2.5) Global visible fuzzy target regions (low-res)
        # Same for all agents (shared prior)
        # -------------------------------------------------
        if hasattr(world, "get_target_regions_lowres"):
            region_low = world.get_target_regions_lowres().astype(np.float32, copy=False).ravel()
            obs_parts.append(region_low)
            obs_debug_parts.append(("region_low", tuple(region_low.shape)))

        # -------------------------------------------------
        # 3) Fixed-size local perceived grid map patch
        # -------------------------------------------------
        PATCH_RADIUS = 7  # grid cells -> (2*7+1)=15, patch dim=225 (reasonable for MLP)
        gx, gy = world.to_grid(pos[0], pos[1])

        # Vectorized patch extraction with padding (outside map treated as unknown=1)
        R = PATCH_RADIUS
        H, W = agent.perceived_grid_map.shape

        x0, x1 = gx - R, gx + R + 1
        y0, y1 = gy - R, gy + R + 1

        # Clip to valid range for slicing
        sx0, sx1 = max(0, x0), min(H, x1)
        sy0, sy1 = max(0, y0), min(W, y1)

        patch_mat = agent.perceived_grid_map[sx0:sx1, sy0:sy1]

        # Padding needed on each side to reach (2R+1, 2R+1)
        pad_left = sx0 - x0
        pad_right = x1 - sx1
        pad_bottom = sy0 - y0
        pad_top = y1 - sy1

        if pad_left or pad_right or pad_bottom or pad_top:
            patch_mat = np.pad(
                patch_mat,
                ((pad_left, pad_right), (pad_bottom, pad_top)),
                mode="constant",
                constant_values=1,
            )

        # -------------------------------------------------
        # 3.5) Merge visible target patch into local perceived patch
        # Three-valued local map:
        #   0 = known free cell
        #   1 = unknown / unobserved / padded area
        #   2 = currently visible unfinished target cell
        # -------------------------------------------------
        if getattr(agent, "perceived_target_map", None) is None:
            agent.perceived_target_map = np.zeros_like(agent.perceived_grid_map, dtype=np.uint8)

        target_patch_mat = agent.perceived_target_map[sx0:sx1, sy0:sy1]

        if pad_left or pad_right or pad_bottom or pad_top:
            target_patch_mat = np.pad(
                target_patch_mat,
                ((pad_left, pad_right), (pad_bottom, pad_top)),
                mode="constant",
                constant_values=0,
            )

        merged_patch_mat = patch_mat.astype(np.float32, copy=True)
        merged_patch_mat[target_patch_mat > 0] = 2.0

        patch = merged_patch_mat.astype(np.float32, copy=False).ravel()
        obs_parts.append(patch)
        obs_debug_parts.append(("local_patch", tuple(patch.shape)))

        # -------------------------------------------------
        # 4) Concatenate into 1D observation
        # -------------------------------------------------
        actor_obs = np.concatenate(obs_parts, axis=0).astype(np.float32)
        global_state = self.global_state(world).astype(np.float32)

        total_obs = np.concatenate([actor_obs, global_state], axis=0)

        if not hasattr(self, "_obs_debug_printed"):
            print("\n[OBS DEBUG] observation components:")
            for name, shape in obs_debug_parts:
                print(f"  - {name}: {shape}")
            print("[OBS DEBUG] total obs shape:", total_obs.shape)
            self._obs_debug_printed = True

        return total_obs

    def done(self, agent, world):
        # Episode ends immediately when all hidden true targets are found
        if hasattr(world, "all_targets_found"):
            return bool(world.all_targets_found())
        return False
