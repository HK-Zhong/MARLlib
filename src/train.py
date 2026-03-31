import os
from marllib import marl
from src.uwb_callback import UWBCustomMetricsCallbacks

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# prepare env
# return: (env, env_config)
env = marl.make_env(environment_name="uwb_planning_env", map_name="Scenario1")

# initialize algorithm with appointed hyperparameters
marl_algo = marl.algos.mappo(hyperparam_source='mpe')

print(type(marl_algo))

# build agent model based on env + algorithms + user preference
# return: (model(deep learning model), model_config)
model = marl.build_model(env, marl_algo,
                         {"core_arch": "mlp", "encode_layer": "256-512"})

# 输出一下model的内容（测试用）

timesteps_total = 900000

# start training
marl_algo.fit(
    env, model,
    stop={'timesteps_total': timesteps_total},
    share_policy='group',
    checkpoint_freq=10,
    checkpoint_at_end=True,
    num_workers=4,
    num_envs_per_worker=2,
    num_gpus=0,
    seed=42,
    evaluation_interval=10,
    evaluation_num_episodes=10,
    evaluation_config={"explore": False},
    callbacks=UWBCustomMetricsCallbacks,
)

# rendering
# mappo.render(
#     env, model,
#     stop={'timesteps_total': 2 * timesteps_total},
#     restore_path={
#         # experiment configuration
#         'params_path': "/home/coolas-fly/MARLlib/exp_results/mappo_mlp_Scenario1/Ours2_lr0.0001/params.json",
#         # checkpoint path
#         'model_path': "/home/coolas-fly/MARLlib/exp_results/mappo_mlp_Scenario1/New/checkpoint_000240/checkpoint-240"},
#         share_policy='group',
#         checkpoint_freq=10,
#         checkpoint_at_end=True,
#         num_workers=4,
#         num_envs_per_worker=2,
#         num_gpus=0,
#         seed=42,
#         evaluation_interval=10,
#         evaluation_num_episodes=10,
#         evaluation_config={"explore": False},
#         callbacks=UWBCustomMetricsCallbacks,
# )
