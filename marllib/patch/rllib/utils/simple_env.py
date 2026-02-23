import numpy as np
from gym import spaces
from gym.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

import os
import json
from datetime import datetime


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env
    return env


class SimpleEnv(AECEnv):
    def __init__(self, scenario, world, max_cycles, continuous_actions=False, local_ratio=None):
        super().__init__()

        self.seed()

        self.metadata = {'render.modes': ['human', 'rgb_array']}

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {agent.name: idx for idx, agent in enumerate(self.world.agents)}

        self._agent_selector = agent_selector(self.agents)

        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 1
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(low=0, high=1, shape=(space_dim,))
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(obs_dim,), dtype=np.float32)

        self.state_space = spaces.Box(low=-np.float32(np.inf), high=+np.float32(np.inf), shape=(state_dim,), dtype=np.float32)

        self.steps = 0

        self.current_actions = [None] * self.num_agents

        self.viewer = None

        # -------------------------------------------------
        # Custom metric file logger (writes JSONL)
        # -------------------------------------------------
        self._custom_metric_log_path = self._init_custom_metric_logger()
        # Episode-grouped buffer (one JSON line per episode)
        self._custom_metric_episode_idx = 0
        self._custom_metric_episode_records = []
        # Step counters for logging
        self._custom_metric_step_in_episode = 0
        self._custom_metric_global_step = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.scenario.observation(self.world.agents[self._index_map[agent]], self.world).astype(np.float32)

    def state(self):
        states = tuple(self.scenario.observation(self.world.agents[self._index_map[agent]], self.world).astype(np.float32) for agent in self.possible_agents)
        return np.concatenate(states, axis=None)

    def reset(self):
        # Flush unfinished episode records (if any) before resetting
        try:
            self._flush_custom_metrics_episode(done=False)
        except Exception:
            pass

        # Reset per-episode step counter on reset
        self._custom_metric_step_in_episode = 0

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0. for name in self.agents}
        self._cumulative_rewards = {name: 0. for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self._reset_render()

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents

    # -------------------------------------------------
    # Custom metric logging (independent of RLlib infos)
    # -------------------------------------------------
    def _find_project_root(self):
        """Try to locate project root containing `exp_results` directory."""
        cur = os.path.abspath(os.path.dirname(__file__))
        for _ in range(10):
            if os.path.isdir(os.path.join(cur, "exp_results")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        # fallback: go up to MARLlib root (best effort)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))

    def _init_custom_metric_logger(self):
        """Create log directory and a new timestamped json file, return its path."""
        root = self._find_project_root()
        log_dir = os.path.join(root, "exp_results", "custom_metric")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(log_dir, f"{ts}.json")
        # Create the file eagerly so users can see it immediately
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass
        return path

    def _collect_custom_metrics_once(self):
        """Collect one metrics record into the in-memory episode buffer."""
        metrics = getattr(self.world, "last_metrics", {}) or {}

        # Step counters (per-episode and global)
        self._custom_metric_step_in_episode = int(getattr(self, "_custom_metric_step_in_episode", 0)) + 1
        self._custom_metric_global_step = int(getattr(self, "_custom_metric_global_step", 0)) + 1

        record = {
            "wall_time": datetime.now().isoformat(timespec="seconds"),
            "env_step": int(self._custom_metric_step_in_episode),
            "global_step": int(self._custom_metric_global_step),
            "completion_ratio": float(metrics.get("completion_ratio", 0.0)),
            "team_new_targets_found": int(metrics.get("team_new_targets_found", 0)),
            "episode_steps_to_done": int(metrics.get("episode_steps_to_done", -1)),
            "first_target_time": int(metrics.get("first_target_time", -1)),
            "last_target_time": int(metrics.get("last_target_time", -1)),
            "role_balance": float(metrics.get("role_balance", 0.0)),
            "overlap_ratio": float(metrics.get("overlap_ratio", 0.0)),
            "collision_count_team": int(metrics.get("collision_count_team", 0)),
            "new_cell_visits_team": int(metrics.get("new_cell_visits_team", 0)),
            "unsafe_ratio": float(metrics.get("unsafe_ratio", 0.0)),
        }

        if not hasattr(self, "_custom_metric_episode_records") or self._custom_metric_episode_records is None:
            self._custom_metric_episode_records = []
        self._custom_metric_episode_records.append(record)

    def _flush_custom_metrics_episode(self, done=False):
        """Flush current episode buffer as ONE dict (one line) into the json file."""
        path = getattr(self, "_custom_metric_log_path", None)
        if not path:
            self._custom_metric_log_path = self._init_custom_metric_logger()
            path = self._custom_metric_log_path

        records = getattr(self, "_custom_metric_episode_records", None) or []
        if len(records) == 0:
            return

        ep_idx = int(getattr(self, "_custom_metric_episode_idx", 0))
        episode_obj = {
            "episode": ep_idx,
            "done": bool(done),
            "num_steps": int(len(records)),
            "records": records,
        }

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode_obj, ensure_ascii=False) + "\n")
        except Exception:
            # Avoid crashing training due to logging failures
            pass

        # Reset buffer for next episode
        self._custom_metric_episode_idx = ep_idx + 1
        self._custom_metric_episode_records = []
        self._custom_metric_step_in_episode = 0

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = global_reward * (1 - self.local_ratio) + agent_reward * self.local_ratio
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.continuous_actions:
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            # Collect custom metrics once per environment step (not per agent)
            self._collect_custom_metrics_once()

            # Apply max-cycles termination first
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.dones[a] = True

            # If episode terminates, flush episode group
            try:
                if all(self.dones.values()):
                    self._flush_custom_metrics_episode(done=True)
            except Exception:
                pass
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

    def render(self, mode='human'):
        from marllib.patch.rllib.utils import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            # from multiagent._mpe_utils import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color[:3], alpha=0.5)
                else:
                    geom.set_color(*entity.color[:3])
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

            self.viewer.text_lines = []
            idx = 0
            for agent in self.world.agents:
                if not agent.silent:
                    tline = rendering.TextLine(self.viewer.window, idx)
                    self.viewer.text_lines.append(tline)
                    idx += 1

        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for idx, other in enumerate(self.world.agents):
            if other.silent:
                continue
            if np.all(other.state.c == 0):
                word = '_'
            elif self.continuous_actions:
                word = '[' + ",".join([f"{comm:.2f}" for comm in other.state.c]) + "]"
            else:
                word = alphabet[np.argmax(other.state.c)]

            message = (other.name + ' sends ' + word + '   ')

            self.viewer.text_lines[idx].set_text(message)

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        self.viewer.set_max_size(cam_range)
        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        # render to display or array
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._reset_render()
