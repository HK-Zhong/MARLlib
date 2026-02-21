import os
import yaml
import numpy as np


class WorldUWBMixIn:
    """
    UWB/LOS related component for UWBPlanningWorld.

    Depends on WorldMapMixin providing:
      - self.grid_map, self.grid_size
      - self.to_grid(), self.to_real()
      - self.map_size_m, self.map_resolution_m
    """

    # -------------------------------
    # UWB members init (called in World.__init__)
    # -------------------------------
    def _init_uwb_members(self):
        # Ground-truth anchors in REAL coords: {id: (x, y)}
        self.uwb_locations = dict()

        # Precomputed anchor grid coordinates: {id: (gx, gy)}
        self.uwb_grid_locations = dict()

        # Vectorized anchor storage (aligned arrays)
        self.uwb_anchor_ids = np.array([], dtype=np.int32)
        self.uwb_anchor_xs = np.array([], dtype=np.float32)
        self.uwb_anchor_ys = np.array([], dtype=np.float32)
        self.uwb_anchor_gxs = np.array([], dtype=np.int32)
        self.uwb_anchor_gys = np.array([], dtype=np.int32)

        # Anchor LOS connectivity prior
        self.uwb_los_lines = [
            (0, 1), (1, 2), (2, 3), (1, 4),
            (2, 5), (3, 6), (3, 8), (0, 7),
            (7, 9), (9, 10), (9, 11), (11, 12),
            (12, 13), (11, 13), (8, 13),
            (4, 5), (5, 6)
        ]

        # Cache for LOS checks: ((gx1,gy1),(gx2,gy2)) -> bool (order-invariant via mirrored keys)
        self._los_cache = {}

        # Precomputed anchor-to-anchor LOS prior grid (shared by all agents)
        self._uwb_los_prior_grid = None

        # Cache for free_expand neighborhood offsets (free_expand -> (dxs, dys))
        self._expand_offsets_cache = {}

    # -------------------------------------------------
    # UWB initialization called from world.map_init()
    # -------------------------------------------------
    def uwb_init(self, anchors_file=None, free_expand_prior=1):
        """
        Load anchors -> build grid coords/arrays -> precompute anchor-anchor LOS prior grid.
        Call this once after obstacles are loaded (grid_map ready).
        """
        self.uwb_locations = self.uwb_anchors_init(anchors_file=anchors_file)

        # Precompute anchor grid coordinates once (for faster per-step queries)
        self.uwb_grid_locations = {
            aid: self.to_grid(xy[0], xy[1]) for aid, xy in self.uwb_locations.items()
        }

        # Build aligned numpy arrays for fast distance filtering
        if self.uwb_locations:
            _ids = np.array(sorted(self.uwb_locations.keys()), dtype=np.int32)
            _xs = np.array([self.uwb_locations[i][0] for i in _ids], dtype=np.float32)
            _ys = np.array([self.uwb_locations[i][1] for i in _ids], dtype=np.float32)
            _gxy = np.array([self.uwb_grid_locations[i] for i in _ids], dtype=np.int32)

            self.uwb_anchor_ids = _ids
            self.uwb_anchor_xs = _xs
            self.uwb_anchor_ys = _ys
            self.uwb_anchor_gxs = _gxy[:, 0]
            self.uwb_anchor_gys = _gxy[:, 1]
        else:
            self.uwb_anchor_ids = np.array([], dtype=np.int32)
            self.uwb_anchor_xs = np.array([], dtype=np.float32)
            self.uwb_anchor_ys = np.array([], dtype=np.float32)
            self.uwb_anchor_gxs = np.array([], dtype=np.int32)
            self.uwb_anchor_gys = np.array([], dtype=np.int32)

        # Precompute anchor-anchor LOS prior grid once
        self._uwb_los_prior_grid = self._build_uwb_los_prior_grid(free_expand=free_expand_prior)

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
            anchors_file = os.path.join(os.path.dirname(__file__), "UWB_Anchors.yml")

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

    # =====================================================
    # Agent perceived-map init using anchor-anchor LOS prior
    # =====================================================
    def uwb_los_prior_to_map(self, agent, free_expand=1):
        """
        Initialize an agent's perceived map using anchor-to-anchor LOS priors.

        This does NOT modify the world ground-truth `self.grid_map`. Instead, it updates
        `agent.perceived_grid_map` in-place by setting cells along anchor-to-anchor LOS
        corridors (defined by `self.uwb_los_lines`) to 0.

        Args:
            agent: The agent whose `perceived_grid_map` will be updated.
            free_expand (int): expand each carved cell to a neighborhood.
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
        if self._uwb_los_prior_grid is None:
            return

        # Merge shared prior grid into agent's perceived map (free = 0)
        target_grid_map[self._uwb_los_prior_grid == 0] = 0

    def _build_uwb_los_prior_grid(self, free_expand=1):
        """
        Precompute a grid map encoding anchor-to-anchor LOS corridors.
        Vectorized carving (no Python nested loops).
        """
        prior_grid = np.ones_like(self.grid_map, dtype=np.int8)

        if not self.uwb_locations:
            return prior_grid

        for a_id, b_id in self.uwb_los_lines:
            if a_id not in self.uwb_grid_locations or b_id not in self.uwb_grid_locations:
                continue

            start = self.uwb_grid_locations[a_id]
            end = self.uwb_grid_locations[b_id]

            # Check LOS using cache
            if not self._has_los_cached(start, end):
                continue

            # Get ray cells once
            _, ray_cells = self._bresenham_ray_cells_and_los(start[0], start[1], end[0], end[1])

            # Vectorized carving
            self._carve_cells(prior_grid, ray_cells, free_expand)

        return prior_grid

    # =====================================================
    # LOS cache + LOS check
    # =====================================================
    def _has_los_cached(self, start_grid, end_grid):
        """
        Cached LOS check between two grid points.
        Order-invariant: (a,b) == (b,a).
        """
        key = (start_grid, end_grid)
        key_rev = (end_grid, start_grid)

        if not hasattr(self, "_los_cache") or self._los_cache is None:
            self._los_cache = {}

        if key in self._los_cache:
            return self._los_cache[key]
        if key_rev in self._los_cache:
            return self._los_cache[key_rev]

        los = self._has_los(start_grid, end_grid)
        self._los_cache[key] = los
        return los

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

    # =====================================================
    # LOS-based belief update: agent <-> anchors
    # =====================================================
    def los_based_map_update(self, agent, update_map=False, free_expand=1):
        """
        Return IDs of UWB anchors within agent perception range AND with LOS to agent.
        If update_map=True, carve LOS rays into agent.perceived_grid_map.

        new_free_count in your current design:
          - counts ONLY when the agent arrives at a new unvisited grid cell
        """
        agent_rx, agent_ry = agent.state.p_pos
        perception_range = float(getattr(agent, "perception_range", 0.0))
        agent_gx, agent_gy = self.to_grid(agent_rx, agent_ry)

        # Ensure perceived map exists if we will update it
        if update_map and getattr(agent, "perceived_grid_map", None) is None:
            agent.perceived_grid_map = np.ones_like(self.grid_map, dtype=np.int8)

        visible_ids = []
        new_free_count = 0

        # visited-grid count logic (your current rule)
        if getattr(agent, "visited_grid_map", None) is None:
            agent.visited_grid_map = np.zeros_like(self.grid_map, dtype=np.int8)

        if 0 <= agent_gx < self.grid_size and 0 <= agent_gy < self.grid_size:
            if agent.visited_grid_map[agent_gx, agent_gy] == 0:
                agent.visited_grid_map[agent_gx, agent_gy] = 1
                new_free_count = 1

        in_range_idx = self._anchors_in_range_indices(agent_rx, agent_ry, perception_range)
        if in_range_idx.size == 0:
            return visible_ids, new_free_count

        for idx in in_range_idx:
            anchor_id = int(self.uwb_anchor_ids[idx])
            anchor_gx = int(self.uwb_anchor_gxs[idx])
            anchor_gy = int(self.uwb_anchor_gys[idx])

            start = (agent_gx, agent_gy)
            end = (anchor_gx, anchor_gy)

            # Fast path: LOS query only
            if not update_map:
                if self._has_los_cached(start, end):
                    visible_ids.append(anchor_id)
                continue

            # -------------------------
            # update_map=True branch (uses cache to avoid duplicate bresenham)
            # -------------------------
            cached = None
            if hasattr(self, "_los_cache") and self._los_cache is not None:
                cached = self._los_cache.get((start, end))
                if cached is None:
                    cached = self._los_cache.get((end, start))

            if cached is False:
                continue

            if cached:
                _, ray_cells = self._bresenham_ray_cells_and_los(agent_gx, agent_gy, anchor_gx, anchor_gy)
                visible_ids.append(anchor_id)
                self._carve_cells(agent.perceived_grid_map, ray_cells, free_expand)
                continue

            los_ok, ray_cells = self._bresenham_ray_cells_and_los(agent_gx, agent_gy, anchor_gx, anchor_gy)
            if not hasattr(self, "_los_cache") or self._los_cache is None:
                self._los_cache = {}
            self._los_cache[(start, end)] = los_ok
            self._los_cache[(end, start)] = los_ok

            if not los_ok:
                continue

            visible_ids.append(anchor_id)
            self._carve_cells(agent.perceived_grid_map, ray_cells, free_expand)

        return visible_ids, new_free_count

    # =====================================================
    # Carving + offsets
    # =====================================================
    def _carve_cells(self, grid_map, ray_cells, free_expand):
        """Carve ray_cells into grid_map (set to 0). Does NOT count newly discovered cells."""
        if not ray_cells:
            return

        cells = np.asarray(ray_cells, dtype=np.int32)
        dxs, dys = self._get_expand_offsets(free_expand)

        nx = cells[:, 0:1] + dxs.reshape(1, -1)
        ny = cells[:, 1:2] + dys.reshape(1, -1)

        nx = nx.ravel()
        ny = ny.ravel()

        mask = (nx >= 0) & (nx < self.grid_size) & (ny >= 0) & (ny < self.grid_size)
        if not np.any(mask):
            return

        nxm = nx[mask].astype(np.int32, copy=False)
        nym = ny[mask].astype(np.int32, copy=False)

        lin = nxm * self.grid_size + nym
        flat = grid_map.ravel()
        flat[lin] = 0

    def _get_expand_offsets(self, free_expand):
        """Return cached neighborhood offsets for a given free_expand."""
        k = int(max(0, free_expand))
        if not hasattr(self, "_expand_offsets_cache") or self._expand_offsets_cache is None:
            self._expand_offsets_cache = {}

        if k in self._expand_offsets_cache:
            return self._expand_offsets_cache[k]

        if k == 0:
            dxs = np.array([0], dtype=np.int32)
            dys = np.array([0], dtype=np.int32)
        else:
            rng = np.arange(-k, k + 1, dtype=np.int32)
            dx_grid, dy_grid = np.meshgrid(rng, rng, indexing="ij")
            dxs = dx_grid.ravel()
            dys = dy_grid.ravel()

        self._expand_offsets_cache[k] = (dxs, dys)
        return dxs, dys

    # =====================================================
    # Helpers
    # =====================================================
    def _anchors_in_range_indices(self, agent_rx, agent_ry, perception_range):
        """Return numpy indices of anchors within perception_range (real coords)."""
        if self.uwb_anchor_ids.size == 0:
            return np.array([], dtype=np.int64)
        dists = np.hypot(self.uwb_anchor_xs - agent_rx, self.uwb_anchor_ys - agent_ry)
        return np.nonzero(dists <= perception_range)[0]

    def _bresenham_ray_cells_and_los(self, x0, y0, x1, y1):
        """Single Bresenham traversal: returns (los_ok, ray_cells)."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        ray_cells = []

        while True:
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                return False, ray_cells
            if self.grid_map[x, y] == 1:
                return False, ray_cells

            ray_cells.append((x, y))

            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return True, ray_cells
