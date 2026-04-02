import os
from marllib import marl
from src.uwb_callback import UWBCustomMetricsCallbacks

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# prepare env
# return: (env, env_config)
env = marl.make_env(environment_name="uwb_planning_env", map_name="Scenario1")

# initialize algorithm with appointed hyperparameters
marl_algo = marl.algos.ippo(hyperparam_source='mpe')

print(type(marl_algo))

# build agent model based on env + algorithms + user preference
# return: (model(deep learning model), model_config)
model = marl.build_model(env, marl_algo,
                         {"core_arch": "mlp", "encode_layer": "256-512"})

# 输出一下model的内容（测试用）

timesteps_total = 900000

# start training
# marl_algo.fit(
#     env, model,
#     stop={'timesteps_total': timesteps_total},
#     share_policy='group',
#     checkpoint_freq=10,
#     checkpoint_at_end=True,
#     num_workers=4,
#     num_envs_per_worker=2,
#     num_gpus=0,
#     seed=42,
#     evaluation_interval=10,
#     evaluation_num_episodes=10,
#     evaluation_config={"explore": False},
#     callbacks=UWBCustomMetricsCallbacks,
# )

# rendering
marl_algo.render(
    env, model,
    stop={'timesteps_total': timesteps_total + 3600 * 30},
    restore_path={
        # experiment configuration
        'params_path': "/home/coolas-fly/MARLlib/exp_results/ippo_mlp_Scenario1/IPPO/params.json",
        # checkpoint path
        'model_path': "/home/coolas-fly/MARLlib/exp_results/ippo_mlp_Scenario1/IPPO/checkpoint_000250/checkpoint-250"},
        share_policy='group',
        checkpoint_freq=5,
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
