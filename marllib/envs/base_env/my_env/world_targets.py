import numpy as np


class WorldTargetsMixIn:
    """
    Mixin class handling:
    - target region generation
    - hidden target point sampling
    - target completion tracking
    - low-resolution region visibility
    - team-level target statistics
    """

    # -------------------------------------------------
    # Target member initialization
    # -------------------------------------------------
    def _init_target_members(self, num_targets=10):
        # number of true hidden targets
        self.num_targets = int(num_targets)

        # one region per target (1:1 mapping)
        self.num_regions = int(self.num_targets)

        # region size = 5% of total map area
        self.region_size_m = 10

        self.region_downsample = 5

        # distance judging whether agent has reached goal
        self.target_reach_dist_m = 1.0

        # target / region containers
        self.target_regions_real = []
        self.target_points_grid = []
        self.target_points_real = []

        self.target_found = []
        self.target_region_ids = []
        self.region_found = []

        self.team_new_targets_found = 0

        self._region_centers = np.zeros((0, 2), dtype=np.float32)

    # -------------------------------------------------
    # Episode-level reset of targets
    # -------------------------------------------------
    def reset_targets(self, rng):
        """
        Generate fuzzy visible regions + hidden true targets.
        One region per target.
        """

        self.num_regions = int(self.num_targets)

        self.target_regions_real = []
        self.target_points_grid = []
        self.target_points_real = []
        self.target_found = []
        self.target_region_ids = []
        self.region_found = []
        self.team_new_targets_found = 0

        self._region_centers = np.zeros((0, 2), dtype=np.float32)

        if hasattr(self, "_target_points_real_array"):
            delattr(self, "_target_points_real_array")

        half = float(self.map_size_m / 2.0)
        region_half = float(self.region_size_m / 2.0)

        def _clamp(v, lo, hi):
            return max(lo, min(hi, v))

        def _sample_free_in_region(xmin, xmax, ymin, ymax, max_tries=4000):
            for _ in range(max_tries):
                if hasattr(rng, "integers"):
                    rx = float(rng.integers(int(xmin), int(xmax) + 1))
                    ry = float(rng.integers(int(ymin), int(ymax) + 1))
                else:
                    rx = float(rng.randint(int(xmin), int(xmax) + 1))
                    ry = float(rng.randint(int(ymin), int(ymax) + 1))

                gx, gy = self.to_grid(rx, ry)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    if self.grid_map[gx, gy] == 0:
                        return (gx, gy), (rx, ry)
            return None, None

        centers_list = []
        max_tries = max(5000, 500 * self.num_targets)
        tries = 0

        while len(self.target_regions_real) < self.num_targets and tries < max_tries:
            tries += 1

            cx = float(rng.uniform(-half + region_half, half - region_half))
            cy = float(rng.uniform(-half + region_half, half - region_half))

            xmin = _clamp(cx - region_half, -half, half)
            xmax = _clamp(cx + region_half, -half, half)
            ymin = _clamp(cy - region_half, -half, half)
            ymax = _clamp(cy + region_half, -half, half)

            (tg, tr) = _sample_free_in_region(xmin, xmax, ymin, ymax)
            if tg is None:
                continue

            rid = len(self.target_regions_real)

            self.target_regions_real.append((xmin, xmax, ymin, ymax))
            centers_list.append((0.5 * (xmin + xmax), 0.5 * (ymin + ymax)))

            self.target_points_grid.append(tg)
            self.target_points_real.append(tr)

            self.target_found.append(False)
            self.target_region_ids.append(rid)
            self.region_found.append(False)

        if len(centers_list) > 0:
            self._region_centers = np.array(centers_list, dtype=np.float32)

        self.num_regions = len(self.target_regions_real)

    # -------------------------------------------------
    # Target arrival detection (called in world.step)
    # -------------------------------------------------
    def update_target_completion(self):
        """Check whether any agent physically reaches a target (team-level).

        Optimization notes:
        - Only checks unfinished targets.
        - Vectorized distance computation (no Python loop over targets).
        - Early-exit when all unfinished targets have been found.

        Distance rule:
        - A target is considered reached when Euclidean distance <= target_reach_dist_m.
        """

        # Per-step counter: how many targets are newly found in this step
        self.team_new_targets_found = 0

        if not self.target_points_grid:
            return

        # Cache target real coordinates (meters)
        if not hasattr(self, "_target_points_real_array"):
            self._target_points_real_array = np.array(
                [self.to_real(gx, gy) for gx, gy in self.target_points_grid],
                dtype=np.float32,
            )

        target_xy = self._target_points_real_array  # (N, 2)

        # Determine unfinished targets
        found = np.asarray(self.target_found, dtype=bool)
        unfinished_idx = np.nonzero(~found)[0]
        if unfinished_idx.size == 0:
            return

        target_xy_u = target_xy[unfinished_idx]  # (M, 2)
        # thr_sq = float(self.target_reach_dist_m) ** 2
        thr_sq = self.target_reach_dist_m

        newly_found_mask = np.zeros((unfinished_idx.size,), dtype=bool)

        # Any agent can complete any target (team-level)
        for agent in self.agents:
            if agent.state.p_pos is None:
                continue

            ax, ay = agent.state.p_pos
            dx = target_xy_u[:, 0] - ax
            dy = target_xy_u[:, 1] - ay
            dist_sq = dx * dx + dy * dy

            newly_found_mask |= (dist_sq <= thr_sq)

            # Early exit if all remaining unfinished targets are already hit
            if newly_found_mask.all():
                break

        if not newly_found_mask.any():
            return

        hit_idx = unfinished_idx[newly_found_mask]

        # Update python lists in-place
        for i in hit_idx.tolist():
            self.target_found[i] = True
            self.region_found[i] = True

        self.team_new_targets_found = int(hit_idx.size)

    # -------------------------------------------------
    # Low-res region visibility
    # -------------------------------------------------
    def get_target_regions_lowres(self):
        ds = max(1, self.region_downsample)
        h = (self.grid_size + ds - 1) // ds
        w = (self.grid_size + ds - 1) // ds

        low = np.zeros((h, w), dtype=np.int8)

        for idx, (xmin, xmax, ymin, ymax) in enumerate(self.target_regions_real):
            if self.region_found[idx]:
                continue

            gx0, gy0 = self.to_grid(xmin, ymin)
            gx1, gy1 = self.to_grid(xmax, ymax)

            min_gx, max_gx = sorted((gx0, gx1))
            min_gy, max_gy = sorted((gy0, gy1))

            min_gx = np.clip(min_gx, 0, self.grid_size - 1)
            max_gx = np.clip(max_gx, 0, self.grid_size - 1)
            min_gy = np.clip(min_gy, 0, self.grid_size - 1)
            max_gy = np.clip(max_gy, 0, self.grid_size - 1)

            lx0, lx1 = min_gx // ds, max_gx // ds
            ly0, ly1 = min_gy // ds, max_gy // ds

            low[lx0:lx1 + 1, ly0:ly1 + 1] = 1

        return low

    # -------------------------------------------------
    # Convenience metrics
    # -------------------------------------------------
    def get_targets_remaining(self):
        """Return the number of unfinished targets."""
        if not self.target_found:
            return int(getattr(self, "num_targets", 0))
        # target_found is a python list of bool
        return int(len(self.target_found) - sum(1 for f in self.target_found if f))

    def get_completion_ratio(self):
        """Return completion ratio in [0, 1]."""
        total = int(getattr(self, "num_targets", 0))
        if total <= 0:
            return 0.0
        if not self.target_found:
            return 0.0
        done = float(sum(1 for f in self.target_found if f))
        return float(done / total)

    # -------------------------------------------------
    # Completion check
    # -------------------------------------------------
    def all_targets_found(self):
        return bool(self.target_found) and all(self.target_found)
