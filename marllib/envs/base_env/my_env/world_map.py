import os
import yaml
import numpy as np
from scipy.ndimage import distance_transform_edt


class WorldMapMixin:
    """
    Map-related component for UWBPlanningWorld.

    Responsibilities:
    - Map geometry: map_size_m, map_resolution_m, grid_size
    - Ground-truth occupancy grid: grid_map (0 free, 1 occupied)
    - EDT distance field: edt_map
    - Coordinate transforms: to_grid / to_real
    - Obstacle loading + rasterization: obstacles_init / _fill_rectangle
    - EDT maintenance + safety query: edt_map_init / is_safe
    - Unified map init (MAP PART ONLY): map_init_obstacles_and_edt
    """

    # -------------------------------
    # Map structure init (called inside World.__init__)
    # -------------------------------
    def _init_map_members(self, map_size=50.0, map_resolution=0.5):
        # Grid map (square occupancy grid)
        self.map_size_m = float(map_size)              # map side length in meters
        self.map_resolution_m = float(map_resolution)  # meters per grid cell
        self.safe_distance_m = 1.0
        self.grid_size = int(self.map_size_m / self.map_resolution_m)

        # 0 = free, 1 = occupied/unknown (ground truth)
        self.grid_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Euclidean Distance Transform (meters), derived from grid_map
        self.edt_map = None

    # -------------------------------------------------
    # Map init (ONLY obstacles + EDT here)
    # -------------------------------------------------
    def map_init_obstacles_and_edt(self, obstacles_file=None):
        """
        Initialize MAP-only components:
        1) Load obstacles and update grid_map
        2) Compute EDT from updated grid_map

        Note:
        - UWB anchors are NOT initialized here (belongs to world_uwb.py).
        """
        if obstacles_file is None:
            obstacles_file = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "obstacles.yaml")
            )

        self.obstacles_init(obstacles_file)
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
        if not obstacles_file or (not os.path.exists(obstacles_file)):
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