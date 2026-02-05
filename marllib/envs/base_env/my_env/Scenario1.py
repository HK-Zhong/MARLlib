import numpy as np
from marllib.envs.base_env.my_env.utils import BaseScenario, Agent, UWBPlanningWorld


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
        # Combine terms
        # -------------------------
        W_INFO = 5.0
        W_VISIBLE = 0.1
        W_COLLIDE = 1.0
        W_UNSAFE = 0.5
        W_STEP = 0.01

        rew = 0.0
        rew += W_INFO * info_gain
        rew += W_VISIBLE * (num_visible / float(total_anchors))
        rew -= W_COLLIDE * float(collisions)
        rew -= W_UNSAFE * float(unsafe)
        rew -= W_STEP

        return float(rew)

    def global_reward(self, world):
        """
        Cooperative exploration global reward (team-level):

        + TEAM information gain: union of newly discovered free cells across agents
        + small shaping: average number of visible UWB anchors
        - inter-agent collisions (pairwise)
        - optional safety penalty (count unsafe agents)
        - small step penalty
        """
        # -------------------------
        # TEAM information gain (incremental, summed over agents)
        # Note: This sums per-agent newly discovered cells and may over-count
        # overlaps if multiple agents discover the same cell in the same step.
        # It is fast and works well as a cooperative shaping signal.
        # -------------------------
        denom = float(max(world.grid_size * world.grid_size, 1))
        team_new_free = 0
        for agent in world.agents:
            team_new_free += int(getattr(agent, "last_new_free_count", 0))

        # Prevent extreme spikes if overlaps occur
        team_new_free = min(team_new_free, int(denom))
        info_gain_team = (float(team_new_free) / denom) if denom > 0 else 0.0

        # -------------------------
        # Visible anchors shaping (average, query only)
        # -------------------------
        total_anchors = max(len(getattr(world, "uwb_locations", {})), 1)
        visible_sum = 0.0
        for agent in world.agents:
            visible_ids = agent.last_visible_uwb_ids
            visible_sum += float(len(visible_ids)) / float(total_anchors)
        visible_avg = visible_sum / float(max(len(world.agents), 1))

        # -------------------------
        # Global collision penalty (pairwise)
        # -------------------------
        collisions = 0
        n = len(world.agents)
        for i in range(n):
            for j in range(i + 1, n):
                if self.is_collision(world.agents[i], world.agents[j]):
                    collisions += 1

        # -------------------------
        # Optional global safety penalty (count unsafe agents)
        # -------------------------
        unsafe_count = 0
        if hasattr(world, "is_safe"):
            for agent in world.agents:
                x_real, y_real = agent.state.p_pos
                if not world.is_safe(float(x_real), float(y_real)):
                    unsafe_count += 1

        # -------------------------
        # Combine terms
        # -------------------------
        W_TEAM_INFO = 8.0
        W_VISIBLE = 0.2
        W_COLLIDE = 1.0
        W_UNSAFE = 0.2
        W_STEP = 0.01

        rew = 0.0
        rew += W_TEAM_INFO * info_gain_team
        rew += W_VISIBLE * visible_avg
        rew -= W_COLLIDE * float(collisions)
        rew -= W_UNSAFE * float(unsafe_count)
        rew -= W_STEP

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

        patch = patch_mat.astype(np.float32, copy=False).ravel()
        obs_parts.append(patch)

        # -------------------------------------------------
        # 4) Concatenate into 1D observation
        # -------------------------------------------------
        return np.concatenate(obs_parts, axis=0)
