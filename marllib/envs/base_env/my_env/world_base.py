import numpy as np
from .world_map import WorldMapMixin
from .world_uwb import WorldUWBMixIn
from .world_targets import WorldTargetsMixIn


class UWBPlanningWorld(WorldMapMixin, WorldUWBMixIn, WorldTargetsMixIn):  # multi-agent world
    """
    UWBPlanningWorld
    ----------------
    A modular multi-agent world that integrates:

    1) WorldMapMixin      → Grid map, obstacles, EDT maintenance
    2) WorldUWBMixIn      → UWB anchors, LOS computation, belief update
    3) WorldTargetsMixIn  → Target regions and hidden goal logic

    Design philosophy:
    - All external interfaces use REAL coordinates (meters).
    - Internally, grid coordinates are used for collision checks and LOS carving.
    - Step logic is simplified: no physical forces, direct velocity control.
    - Target detection and belief update are modularized via mixins.
    """

    def __init__(self, map_size=50.0, map_resolution=0.5):
        # =====================================================
        # Core simulation attributes (inherited MPE-style state)
        # =====================================================
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.2

        # =====================================================
        # Grid map / obstacle representation (WorldMapMixin)
        # =====================================================
        self._init_map_members(map_size, map_resolution)

        # =====================================================
        # UWB anchors & LOS logic (WorldUWBMixIn)
        # =====================================================
        self._init_uwb_members()

        # =====================================================
        # Target regions & hidden goals (WorldTargetsMixIn)
        # =====================================================
        self._init_target_members(num_targets=10)

        # unified map initialization
        self.map_init()

    # -------------------------------------------------
    # Unified map initialization (obstacles, EDT, UWB)
    # -------------------------------------------------
    def map_init(self):
        """
        Initialize map-related components in deterministic order:

        1) Load obstacles and compute EDT (ground truth map).
        2) Load UWB anchor configuration and precompute LOS priors.

        This function should be called once during world construction.
        """
        # 1) Load obstacles and update grid_map
        self.map_init_obstacles_and_edt()

        # 2) load UWB anchors + precompute arrays + build anchor-anchor LOS prior grid
        self.uwb_init(anchors_file=None, free_expand_prior=1)

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def reset(self, np_random=None):
        """
        Reset world state for a new episode.

        Operations performed:
        1) Randomly sample valid agent positions (real coordinates).
        2) Reset velocities and action buffers.
        3) Initialize each agent's perceived map to unknown.
        4) Apply anchor-anchor LOS prior.
        5) Perform initial agent-anchor LOS update.
        6) Regenerate target regions and hidden targets.

        Notes:
        - External interface positions are REAL coordinates (meters).
        - Grid coordinates are only used internally.
        """
        rng = np_random if np_random is not None else np.random

        # -------------------------------------------------
        # 1) sample initial real positions (integers) and reset physical/action states
        # -------------------------------------------------
        half = int(self.map_size_m / 2.0)

        def _is_free_real(real_x, real_y):
            grid_x, grid_y = self.to_grid(real_x, real_y)
            if grid_x < 0 or grid_x >= self.grid_size or grid_y < 0 or grid_y >= self.grid_size:
                return False
            return self.grid_map[grid_x, grid_y] == 0

        # precompute a fallback free position (real coords) if needed
        fallback_rx, fallback_ry = 0, 0
        if not _is_free_real(fallback_rx, fallback_ry):
            found = False
            for gx in range(self.grid_size):
                for gy in range(self.grid_size):
                    if self.grid_map[gx, gy] == 0:
                        fallback_rx, fallback_ry = self.to_real(gx, gy)
                        found = True
                        break
                if found:
                    break

        for agent in self.agents:
            # reset velocities
            agent.state.p_vel = np.zeros(self.dim_p, dtype=np.float32)

            # reset actions (velocity command + comm)
            agent.action.u = np.zeros(self.dim_p, dtype=np.float32)
            if self.dim_c > 0:
                agent.action.c = np.zeros(self.dim_c, dtype=np.float32)
                agent.state.c = np.zeros(self.dim_c, dtype=np.float32)
            else:
                agent.action.c = np.zeros((0,), dtype=np.float32)
                agent.state.c = np.zeros((0,), dtype=np.float32)

            # sample a valid position
            placed = False
            for _ in range(10000):
                if hasattr(rng, "integers"):
                    rx = int(rng.integers(-half, half + 1))
                    ry = int(rng.integers(-half, half + 1))
                else:
                    rx = int(rng.randint(-half, half + 1))
                    ry = int(rng.randint(-half, half + 1))
                if _is_free_real(rx, ry):
                    agent.state.p_pos = np.array([rx, ry], dtype=np.float32)
                    placed = True
                    break

            if not placed:
                agent.state.p_pos = np.array([fallback_rx, fallback_ry], dtype=np.float32)

        # -------------------------------------------------
        # 2) reset each agent's perceived map (belief) to unknown + apply anchor-to-anchor LOS prior
        #    initial LOS update: agent <-> anchors within perception range
        # -------------------------------------------------
        for agent in self.agents:
            agent.perceived_grid_map = np.ones_like(self.grid_map, dtype=np.int8)
            agent.visited_grid_map = np.zeros_like(self.grid_map, dtype=np.int8)
            agent.prev_perceived_grid_map = None
            self.uwb_los_prior_to_map(agent, free_expand=1)
            agent.last_visible_uwb_ids, agent.last_new_free_count = self.los_based_map_update(
                agent, update_map=True, free_expand=1)

        # -------------------------------------------------
        # 3) (Re)generate fuzzy target regions + hidden true targets for this episode
        # -------------------------------------------------
        self.reset_targets(rng=rng)

    def step(self):
        """
        Simplified simulation step (velocity-control mode).

        Dynamics assumptions:
        - No inter-agent collision forces.
        - No obstacle contact forces.
        - action.u is interpreted directly as desired velocity.

        Step sequence:
        1) Apply scripted agent callbacks (if any).
        2) Integrate velocity-based motion.
        3) Clamp positions to map boundaries.
        4) Update target completion (team-level).
        5) Update belief maps via UWB LOS observations.
        """

        # -------------------------------------------------
        # 1) Apply scripted agent policies (if defined)
        # -------------------------------------------------
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # -------------------------------------------------
        # 2) Velocity-based motion integration (no physics forces)
        # -------------------------------------------------
        for agent in self.agents:
            if not agent.movable:
                continue

            # Desired velocity directly from action
            if agent.action.u is None:
                v = np.zeros(self.dim_p, dtype=np.float32)
            else:
                v = np.array(agent.action.u, dtype=np.float32)

            # Optional motor noise
            if agent.u_noise:
                v += np.random.randn(*v.shape) * agent.u_noise

            # Clamp speed
            if agent.max_speed is not None:
                speed = float(np.hypot(v[0], v[1]))
                if speed > agent.max_speed and speed > 1e-9:
                    v = v / speed * agent.max_speed

            # Integrate state
            agent.state.p_vel = v
            if agent.state.p_pos is None:
                agent.state.p_pos = np.zeros(self.dim_p, dtype=np.float32)
            agent.state.p_pos += agent.state.p_vel * self.dt

        # -------------------------------------------------
        # 3) Enforce map boundary constraints (real coordinates)
        # -------------------------------------------------
        half = self.map_size_m / 2.0
        for agent in self.agents:
            if agent.state.p_pos is None:
                continue
            agent.state.p_pos[0] = np.clip(agent.state.p_pos[0], -half, half)
            agent.state.p_pos[1] = np.clip(agent.state.p_pos[1], -half, half)

        # -------------------------------------------------
        # 4) Update target completion status (team-level)
        # -------------------------------------------------
        self.update_target_completion()

        # -------------------------------------------------
        # 5) Belief update via UWB LOS observations
        # -------------------------------------------------
        for agent in self.agents:
            if getattr(agent, "perceived_grid_map", None) is None:
                agent.perceived_grid_map = np.ones_like(self.grid_map, dtype=np.int8)

            agent.last_visible_uwb_ids, agent.last_new_free_count = self.los_based_map_update(
                agent, update_map=True, free_expand=1)
