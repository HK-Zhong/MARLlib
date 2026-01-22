import numpy as np
import os
import yaml


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


class World:  # multi-agent world
    def __init__(self, map_size_m=50.0, map_resolution_m=0.5):
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
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        # -------------------------------
        # Grid map (square occupancy grid)
        # -------------------------------
        self.map_size_m = float(map_size_m)              # map side length in meters
        self.map_resolution_m = float(map_resolution_m)  # meters per grid cell

        self.grid_size = int(self.map_size_m / self.map_resolution_m)

        # 0 = free, 1 = occupied / unknown
        self.grid_map = np.zeros(
            (self.grid_size, self.grid_size), dtype=np.int8
        )
        # Euclidean Distance Transform (meters), derived from grid_map
        self.edt_map = None

        # UWB anchor grid locations (gx, gy) loaded from UWB_Anchors.yml
        self.uwb_locations = []
        self.load_uwb_anchors()
        # obstacle grid locations loaded from YAML
        self.obstacles_grid = []

    # =====================================================
    # EDT maintenance (derived from grid_map)
    # =====================================================
    def update_edt(self):
        """
        Compute EDT from grid_map.
        edt_map unit: meters
        grid_map convention:
          0 = free
          1 = occupied / unknown
        """
        try:
            from scipy.ndimage import distance_transform_edt
        except Exception as e:
            raise ImportError(
                "scipy is required for EDT. Please install scipy (e.g., `pip install scipy`)."
            ) from e

        obstacle_mask = (self.grid_map == 1)
        dist_cells = distance_transform_edt(~obstacle_mask)
        self.edt_map = dist_cells * self.map_resolution_m

    def is_safe(self, gx, gy, safe_distance_m):
        """
        Safety query using EDT.
        Returns True if the EDT distance at (gx, gy) >= safe_distance_m.
        """
        if self.edt_map is None:
            return False
        if not (0 <= gx < self.grid_size and 0 <= gy < self.grid_size):
            return False
        return self.edt_map[gx, gy] >= safe_distance_m

    # =====================================================
    # Coordinate transform (real <-> grid)
    # =====================================================
    def to_grid(self, x_real, y_real):
        """
        Convert real-world coordinates (meters) to grid indices (gx, gy).
        Map is a square centered at (0,0) spanning [-map_size_m/2, +map_size_m/2].
        """
        gx = int((float(x_real) + self.map_size_m / 2.0) / self.map_resolution_m)
        gy = int((float(y_real) + self.map_size_m / 2.0) / self.map_resolution_m)
        return gx, gy

    def to_real(self, gx, gy):
        """
        Convert grid indices (gx, gy) back to real-world coordinates (meters).
        """
        x = int(gx) * self.map_resolution_m - self.map_size_m / 2.0
        y = int(gy) * self.map_resolution_m - self.map_size_m / 2.0
        return round(x, 2), round(y, 2)

    # =====================================================
    # UWB anchors loading (real -> grid locations)
    # =====================================================
    def load_uwb_anchors(self, anchors_file=None):
        """
        Load UWB anchors from YAML config and populate `self.uwb_locations` with
        each anchor's grid coordinate (gx, gy), computed from anchor (x, y).
        YAML format:
          UWB_Anchors:
            - {id: 0, x: ..., y: ..., z: ...}
            - ...
        """
        # Default path: same folder as this utils.py file, named 'UWB_Anchors.yml'
        if anchors_file is None:
            anchors_file = os.path.join(os.path.dirname(__file__), "UWB_Anchors.yml")

        self.uwb_locations = []

        if not os.path.exists(anchors_file):
            return self.uwb_locations

        with open(anchors_file, "r") as f:
            data = yaml.safe_load(f) or {}

        anchors = data.get("UWB_Anchors", [])
        for a in anchors:
            try:
                x = float(a["x"])
                y = float(a["y"])
            except Exception:
                continue
            gx, gy = self.to_grid(x, y)
            self.uwb_locations.append((gx, gy))

        return self.uwb_locations

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

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
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
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # =====================================================
    # Obstacle loading (rectangle obstacles)
    # =====================================================
    def load_obstacles(self, obstacles_file):
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
                    self.obstacles_grid.append((gx, gy))