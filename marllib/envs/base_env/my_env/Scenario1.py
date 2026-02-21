import numpy as np
from marllib.envs.base_env.my_env.world_base import UWBPlanningWorld
from marllib.envs.base_env.my_env.mpe_core import BaseScenario, Agent



class Scenario(BaseScenario):
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

        return world

    def reset_world(self, world, np_random):
        world.reset(np_random)

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

    def reward(self, agent, world):
        """
        Sample reward (minimal, intended to be easy to iterate on):

        + information gain: newly discovered free cells in agent.perceived_grid_map
          (incremental count from world.los_based_map_update)
        + small shaping: more visible UWB anchors is slightly better
        - collisions
        - optional safety penalty (via world.is_safe)
        - small step penalty (encourage efficiency)
        """
        # -------------------------
        # number of visible anchors
        # -------------------------
        visible_ids = getattr(agent, "last_visible_uwb_ids", [])
        num_visible = len(visible_ids)
        total_anchors = max(len(getattr(world, "uwb_locations", {})), 1)

        # -------------------------
        # information gain (incremental, no full-map copy)
        # -------------------------
        new_free = int(getattr(agent, "last_new_free_count", 0))
        denom = float(max(world.grid_size * world.grid_size, 1))
        info_gain = (float(new_free) / denom) if denom > 0 else 0.0

        # -------------------------
        # Collision penalty
        # -------------------------
        collisions = 0
        if getattr(agent, "collide", False):
            for other_agent in world.agents:
                if other_agent is agent:
                    continue
                if self.is_collision(other_agent, agent):
                    collisions += 1

        # -------------------------
        # Optional safety penalty (EDT-based)
        # -------------------------
        unsafe = 0
        if hasattr(world, "is_safe"):
            x_real, y_real = agent.state.p_pos
            if not world.is_safe(float(x_real), float(y_real)):
                unsafe = 1

        # -------------------------
        # Target completion shaping (team-level signal)
        # -------------------------
        team_new_targets = int(getattr(world, "team_new_targets_found", 0))
        all_found = bool(world.all_targets_found()) if hasattr(world, "all_targets_found") else False

        # -------------------------
        # Region proximity shaping (dense guidance)
        # Vectorized using cached world._region_centers (real coords)
        # -------------------------
        region_bonus = 0.0
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
                min_dist = float(np.sqrt(dist_sq.min()))

                # normalize by map diagonal
                max_dist = float(np.sqrt(2.0) * (world.map_size_m / 2.0))
                region_bonus = 1.0 - min_dist / max(max_dist, 1e-6)

        # -------------------------
        # Optimized weights (Version A: task-driven)
        # -------------------------
        W_INFO = 0.5  # exploration reduced
        W_VISIBLE = 0.1
        W_COLLIDE = 1.0
        W_UNSAFE = 0.5
        W_STEP = 0.01
        W_TARGET = 8.0  # stronger target discovery
        W_DONE_BONUS = 80.0  # strong completion bonus
        W_REGION = 1.0  # dense guidance toward target region

        rew = 0.0
        rew += W_INFO * info_gain
        rew += W_VISIBLE * (num_visible / float(total_anchors))
        rew += W_REGION * float(region_bonus)
        rew -= W_COLLIDE * float(collisions)
        rew -= W_UNSAFE * float(unsafe)
        rew -= W_STEP
        rew += W_TARGET * float(team_new_targets)
        if all_found:
            rew += W_DONE_BONUS

        return float(rew)

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
        # Cooperative weights
        # -------------------------
        W_TARGET = 15.0  # strong shared signal for discovering targets
        W_DONE_BONUS = 150.0  # strong terminal reward

        rew = 0.0
        rew += W_TARGET * float(team_new_targets)
        if all_found:
            rew += W_DONE_BONUS

        return float(rew)

    def observation(self, agent, world):
        """
        Sample observation design (normalized + fixed patch).

        Observation = [
            normalized agent position (2),          in [-1, 1] roughly
            normalized agent velocity (2),          scaled by a constant(max velocity)
            normalized number of visible anchors (1),
            flattened local perceived grid map patch ( (2R+1) x (2R+1) )
        ]
        """

        # -------------------------------------------------
        # 1) Update perceived map using UWB LOS (belief update)
        # -------------------------------------------------
        visible_ids = agent.last_visible_uwb_ids
        num_visible = len(visible_ids)

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

        # visible anchors normalization: scale by total anchors (avoid magnitude mismatch)
        total_anchors = max(len(getattr(world, "uwb_locations", {})), 1)
        num_visible_norm = np.array([num_visible / float(total_anchors)], dtype=np.float32)

        obs_parts = [pos_norm, vel_norm, num_visible_norm]

        # -------------------------------------------------
        # 2.5) Global visible fuzzy target regions (low-res)
        # Same for all agents (shared prior)
        # -------------------------------------------------
        if hasattr(world, "get_target_regions_lowres"):
            region_low = world.get_target_regions_lowres().astype(np.float32, copy=False).ravel()
            obs_parts.append(region_low)

        # -------------------------------------------------
        # 3) Fixed-size local perceived grid map patch
        # -------------------------------------------------
        PATCH_RADIUS = 5  # grid cells -> (2*7+1)=15, patch dim=225 (reasonable for MLP)
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

        patch = patch_mat.astype(np.float32, copy=False).ravel()
        obs_parts.append(patch)

        # -------------------------------------------------
        # 4) Concatenate into 1D observation
        # -------------------------------------------------
        return np.concatenate(obs_parts, axis=0)

    def done(self, agent, world):
        # Episode ends immediately when all hidden true targets are found
        if hasattr(world, "all_targets_found"):
            return bool(world.all_targets_found())
        return False
