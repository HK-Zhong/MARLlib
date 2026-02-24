from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
# from marllib.patch.rllib.utils.simple_env import SimpleEnv, make_env
from marllib.envs.base_env.my_env.Scenario1 import Scenario


class RawEnv(SimpleEnv):
    def __init__(self, agent_num=3, local_ratio=0.5, max_cycles=200, continuous_actions=False):
        assert 0. <= local_ratio <= 1., "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(agent_num)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata["name"] = "uwb_planning_env"

        # 这两个是你自己记录用的（不影响 RLlib 的 env/episode 统计）
        self._raw_episode = 0
        self._raw_step_in_episode = 0

    def reset(self, seed=None, options=None):
        # SimpleEnv(AEC) reset 一般也是不返回 info（parallel wrapper 会读取 observe）
        out = super().reset()

        self._raw_episode += 1
        self._raw_step_in_episode = 0

        # 初始化 infos（确保每个 agent 都有 dict）
        for aid in getattr(self, "agents", []):
            if aid not in self.infos or not isinstance(self.infos.get(aid), dict):
                self.infos[aid] = {}

        return out

    def step(self, action):
        """
        AEC env step: 不返回 (obs, rew, done, info)，只更新内部状态。
        parallel_wrapper 会在外面调用 last()/observe() 并最终返回四元组。
        """
        # 1) 先让底层环境正常推进一步（更新 rewards/dones/infos 等）
        super().step(action)

        # 2) 更新你自己的计数（可选）
        self._raw_step_in_episode += 1

        # 3) 取 team-level 指标（你之前约定 Scenario/global_reward 每步会写 world.last_metrics）
        team_metrics = {}
        if hasattr(self.world, "last_metrics") and isinstance(self.world.last_metrics, dict):
            team_metrics = dict(self.world.last_metrics)

        # 可选：写入你自己的 step/episode（只是辅助 debug）
        team_metrics.setdefault("raw_episode", int(self._raw_episode))
        team_metrics.setdefault("raw_env_step", int(self._raw_step_in_episode))

        # 4) 关键：infos 的 key 必须是 agent 子集，绝对不要 '__all__'
        if isinstance(self.infos, dict) and "__all__" in self.infos:
            self.infos.pop("__all__", None)

        # 5) 把 team_metrics 注入每个 agent 的 info（每个 agent 都一样）
        for aid in getattr(self, "agents", []):
            if aid not in self.infos or not isinstance(self.infos.get(aid), dict):
                self.infos[aid] = {}
            # 建议统一塞到 team_metrics 这个字段下，方便 callback 取
            self.infos[aid]["team_metrics"] = team_metrics

env = make_env(RawEnv)
parallel_env = parallel_wrapper_fn(env)
