import numpy as np
import os
import yaml
from scipy.ndimage import distance_transform_edt


class BaseScenario:  # defines scenario upon which the world is built
    def make_world(self):  # create elements of the world
        raise NotImplementedError()

    def reset_world(self, world, np_random):  # create initial conditions of the world
        raise NotImplementedError()


class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class AgentState(EntityState):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance
        self.c = None


class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()


class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

        # agent perception range
        self.perception_range = 5
        self.max_speed = 3.0

        # agent-specific perceived occupancy grid (initialized in UWBPlanningWorld)
        self.perceived_grid_map = None
        self.prev_perceived_grid_map = None
        self.last_visible_uwb_ids = []


class UWBPlanningWorld:  # multi-agent world
    def __init__(self, map_size=50.0, map_resolution=0.5):
        # -------------------------------
        # Original world attributes
        # -------------------------------
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
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        # -------------------------------
        # Grid map (square occupancy grid)
        # -------------------------------
        self.map_size_m = float(map_size)  # map side length in meters
        self.map_resolution_m = float(map_resolution)  # meters per grid cell
        self.safe_distance_m = 1.0
        self.grid_size = int(self.map_size_m / self.map_resolution_m)

        # 0 = free, 1 = occupied / unknown， ground truth
        self.grid_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        # Euclidean Distance Transform (meters), derived from grid_map
        self.edt_map = None

        self.uwb_locations = dict()

        # Anchor LOS connectivity prior
        self.uwb_los_lines = [
            (0, 1), (1, 2), (2, 3), (1, 4),
            (2, 5), (3, 6), (3, 8), (0, 7),
            (7, 9), (9, 10), (9, 11), (11, 12),
            (12, 13), (11, 13), (8, 13),
            (4, 5), (5, 6)
        ]

        # unified map initialization
        self.map_init()

    # -------------------------------------------------
    # Unified map initialization (obstacles, EDT, UWB)
    # -------------------------------------------------
    def map_init(self):
        """
        Initialize all map-related components in a unified order:
        1) Load obstacles and update grid_map
        2) Load UWB anchor ground-truth locations (real coordinates)
        3) Compute EDT map from updated grid_map
        """
        # 1) load obstacle grid locations loaded from YAML, update self.grid_map
        obstacles_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "obstacles.yaml"))
        self.obstacles_init(obstacles_file)

        # 2) load UWB anchor locations (ground truth, real coords)
        # UWB anchors: {anchor_id: (x, y)} in real-world coordinates
        self.uwb_locations = self.uwb_anchors_init()

        # 3) compute EDT from updated grid_map
        self.edt_map_init()

    # =====================================================
    # EDT maintenance (derived from grid_map)
    # =====================================================
    def edt_map_init(self):
        """
        Compute EDT from grid_map.
        edt_map unit: meters
        grid_map convention:
          0 = free
          1 = occupied / unknown
        """
        obstacle_mask = (self.grid_map == 1)
        dist_cells = distance_transform_edt(~obstacle_mask)
        self.edt_map = dist_cells * self.map_resolution_m

    def is_safe(self, x_real, y_real):
        """
        Safety query using EDT.
        Returns True if the EDT distance at (gx, gy) >= safe_distance_m.
        """
        if self.edt_map is None:
            return False

        gx, gy = self.to_grid(x_real, y_real)

        if not (0 <= gx < self.grid_size and 0 <= gy < self.grid_size):
            return False

        return self.edt_map[gx, gy] >= self.safe_distance_m

    # =====================================================
    # Coordinate transform (real <-> grid)
    # =====================================================
    def to_grid(self, x_real, y_real):
        gx = int((x_real + self.map_size_m / 2.0) / self.map_resolution_m)
        gy = int((y_real + self.map_size_m / 2.0) / self.map_resolution_m)
        return gx, gy

    def to_real(self, gx, gy):
        x = gx * self.map_resolution_m - self.map_size_m / 2.0
        y = gy * self.map_resolution_m - self.map_size_m / 2.0
        return round(x, 2), round(y, 2)

    # =====================================================
    # UWB anchors loading (ground-truth, real coordinates)
    # =====================================================
    def uwb_anchors_init(self, anchors_file=None):
        """
        Load UWB anchors from YAML and return a dictionary:
            {anchor_id: (x, y)}
        Coordinates are REAL-WORLD (meters).
        """
        if anchors_file is None:
            anchors_file = os.path.join(
                os.path.dirname(__file__), "UWB_Anchors.yml"
            )

        uwb_dict = {}

        if not os.path.exists(anchors_file):
            return uwb_dict

        with open(anchors_file, "r") as f:
            data = yaml.safe_load(f) or {}

        anchors = data.get("UWB_Anchors", [])
        for a in anchors:
            try:
                aid = int(a["id"])
                x = float(a["x"])
                y = float(a["y"])
            except Exception:
                continue

            uwb_dict[aid] = (x, y)

        return uwb_dict

    def uwb_los_prior_to_map(self, agent, free_expand=1):
        """Initialize an agent's perceived map using anchor-to-anchor LOS priors.

        This does NOT modify the world ground-truth `self.grid_map`. Instead, it updates
        `agent.perceived_grid_map` in-place by setting cells along anchor-to-anchor LOS
        corridors (defined by `self.uwb_los_lines`) to 0.

        LOS is validated against the world ground-truth obstacles (`self.grid_map`) via `_has_los`.

        Args:
            agent: The agent whose `perceived_grid_map` will be updated.
            free_expand (int): Expand each carved cell to a
                (2*free_expand+1)x(2*free_expand+1) neighborhood.
        """
        if agent is None:
            return

        target_grid_map = getattr(agent, "perceived_grid_map", None)
        if target_grid_map is None:
            return
        if not hasattr(target_grid_map, "shape"):
            return
        if target_grid_map.shape != self.grid_map.shape:
            return
        if not self.uwb_locations:
            return

        for a_id, b_id in getattr(self, "uwb_los_lines", []):
            if a_id not in self.uwb_locations or b_id not in self.uwb_locations:
                continue

            a_rx, a_ry = self.uwb_locations[a_id]
            b_rx, b_ry = self.uwb_locations[b_id]

            a_gx, a_gy = self.to_grid(a_rx, a_ry)
            b_gx, b_gy = self.to_grid(b_rx, b_ry)

            # Only carve if LOS is not blocked by obstacles in current ground-truth map
            if not self._has_los((a_gx, a_gy), (b_gx, b_gy)):
                continue

            # Bresenham line traversal from (a_gx,a_gy) to (b_gx,b_gy)
            x0, y0 = a_gx, a_gy
            x1, y1 = b_gx, b_gy
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            x, y = x0, y0
            while True:
                for ex in range(-free_expand, free_expand + 1):
                    for ey in range(-free_expand, free_expand + 1):
                        nx, ny = x + ex, y + ey
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            target_grid_map[nx, ny] = 0

                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

    # =====================================================
    # Obstacle loading (rectangle obstacles)
    # =====================================================

    def obstacles_init(self, obstacles_file):
        """
        Load rectangular obstacles from YAML and mark them in grid_map.

        YAML format:
          obstacles:
            - id: rect_1
              type: rectangle
              corners: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        Coordinates are REAL-WORLD (meters).
        """
        if not os.path.exists(obstacles_file):
            return

        with open(obstacles_file, "r") as f:
            data = yaml.safe_load(f) or {}

        obstacles = data.get("obstacles", [])
        for obs in obstacles:
            if obs.get("type", "") != "rectangle":
                continue
            corners = obs.get("corners", [])
            if len(corners) != 4:
                continue

            # convert corners to grid coordinates
            grid_corners = [self.to_grid(c[0], c[1]) for c in corners]
            self._fill_rectangle(grid_corners)

    def _fill_rectangle(self, grid_corners):
        """
        Fill a rectangle in grid_map given 4 grid corners.
        """
        xs = [c[0] for c in grid_corners]
        ys = [c[1] for c in grid_corners]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        for gx in range(min_x, max_x + 1):
            for gy in range(min_y, max_y + 1):
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    self.grid_map[gx, gy] = 1

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
        """Reset all agents' states and beliefs.

        Notes:
        - All interface-level positions are REAL coordinates (meters).
        - Internally we use grid coordinates for obstacle checks.
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
            agent.prev_perceived_grid_map = None
            self.uwb_los_prior_to_map(agent, free_expand=1)
            agent.last_visible_uwb_ids = self.los_based_map_update(agent, update_map=True, free_expand=0)

    # update state of the world
    def step(self):
        # -------------------------------------------------
        # 1. Set actions for scripted agents (if any)
        # -------------------------------------------------
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # -------------------------------------------------
        # 2. Compute collision forces BETWEEN AGENTS ONLY
        # -------------------------------------------------
        p_force = [None] * len(self.agents)

        for i, agent_a in enumerate(self.agents):
            for j, agent_b in enumerate(self.agents):
                if j <= i:
                    continue
                f_a, f_b = self.get_collision_force(agent_a, agent_b)

                if f_a is not None:
                    if p_force[i] is None:
                        p_force[i] = np.zeros(self.dim_p)
                    p_force[i] += f_a

                if f_b is not None:
                    if p_force[j] is None:
                        p_force[j] = np.zeros(self.dim_p)
                    p_force[j] += f_b

        # -------------------------------------------------
        # 3. Velocity-control dynamics (agents only)
        # -------------------------------------------------
        for i, agent in enumerate(self.agents):
            if not agent.movable:
                continue

            # Desired velocity from action
            if agent.action.u is None:
                v = np.zeros(self.dim_p, dtype=np.float32)
            else:
                v = np.array(agent.action.u, dtype=np.float32)

            # Optional motor noise
            if agent.u_noise:
                v += np.random.randn(*v.shape) * agent.u_noise

            # Collision response as velocity correction
            if p_force[i] is not None:
                v += (p_force[i] / agent.mass) * self.dt

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
        # 4. Clamp agent positions to map bounds (real coords)
        # -------------------------------------------------
        half = self.map_size_m / 2.0
        for agent in self.agents:
            if agent.state.p_pos is None:
                continue
            agent.state.p_pos[0] = np.clip(agent.state.p_pos[0], -half, half)
            agent.state.p_pos[1] = np.clip(agent.state.p_pos[1], -half, half)

        # -------------------------------------------------
        # 5. Belief update: UWB LOS -> perceived_grid_map
        # -------------------------------------------------
        for agent in self.agents:
            if getattr(agent, "perceived_grid_map", None) is None:
                agent.perceived_grid_map = np.ones_like(self.grid_map, dtype=np.int8)

            agent.prev_perceived_grid_map = agent.perceived_grid_map.copy()

            agent.last_visible_uwb_ids = self.los_based_map_update(agent, update_map=True, free_expand=0)

        # -------------------------------------------------
        # 6. Update agent communication state
        # -------------------------------------------------
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if (b <= a):
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if (f_a is not None):
                    if (p_force[a] is None):
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if (f_b is not None):
                    if (p_force[b] is None):
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(
                        np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))

        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size

        # avoid numerical issues (exact overlap)
        if dist < 1e-8:
            # apply no force if perfectly overlapping to avoid NaN
            return [None, None]

        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # =====================================================
    # UWB LOS query within agent perception range
    # =====================================================
    def los_based_map_update(self, agent, update_map=False, free_expand=1):
        """
        Return IDs of UWB anchors that are within the agent's perception range
        AND have line-of-sight (LOS) to the agent.

        Inputs are taken directly from the agent:
          - agent.state.p_pos -> (agent_rx, agent_ry)
          - agent.perception_range -> perception_range

        Args:
            agent: agent object with `state.p_pos`, `perception_range`, and optionally `perceived_grid_map`.
            update_map (bool): if True, carve free cells (set to 0) along LOS rays into `agent.perceived_grid_map`.
            free_expand (int): when update_map is True, expand each carved cell to a
                (2*free_expand+1)x(2*free_expand+1) neighborhood.

        Returns:
            list[int]: IDs of anchors with LOS within perception range
        """
        agent_rx, agent_ry = agent.state.p_pos
        perception_range = getattr(agent, "perception_range", 0.0)

        agent_gx, agent_gy = self.to_grid(agent_rx, agent_ry)

        # --- if we need to update perceived map, ensure it exists ---
        if update_map:
            if getattr(agent, "perceived_grid_map", None) is None:
                agent.perceived_grid_map = np.ones_like(self.grid_map, dtype=np.int8)

        visible_ids = []

        for anchor_id, (anchor_rx, anchor_ry) in self.uwb_locations.items():
            # distance check in real-world coordinates
            dist = np.hypot(anchor_rx - agent_rx, anchor_ry - agent_ry)
            if dist > perception_range:
                continue

            # LOS check in grid space (ground truth)
            anchor_gx, anchor_gy = self.to_grid(anchor_rx, anchor_ry)
            if not self._has_los((agent_gx, agent_gy), (anchor_gx, anchor_gy)):
                continue

            visible_ids.append(anchor_id)

            # optional: update agent perceived map along LOS ray
            if update_map:
                x0, y0 = agent_gx, agent_gy
                x1, y1 = anchor_gx, anchor_gy

                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                err = dx - dy

                x, y = x0, y0
                while True:
                    for dx2 in range(-free_expand, free_expand + 1):
                        for dy2 in range(-free_expand, free_expand + 1):
                            nx, ny = x + dx2, y + dy2
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                agent.perceived_grid_map[nx, ny] = 0

                    if x == x1 and y == y1:
                        break
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x += sx
                    if e2 < dx:
                        err += dx
                        y += sy

        return visible_ids

    def _has_los(self, start_grid, end_grid):
        """
        Check LOS between two grid points using Bresenham line traversal.
        LOS is True if all traversed cells are free (grid_map == 0).
        """
        x0, y0 = start_grid
        x1, y1 = end_grid

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                return False
            if self.grid_map[x, y] == 1:
                return False
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return True
