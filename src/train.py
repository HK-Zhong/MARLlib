import os
from marllib import marl
from src.uwb_callback import UWBCustomMetricsCallbacks

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# prepare env
# return: (env, env_config)
env = marl.make_env(environment_name="uwb_planning_env", map_name="Scenario1")

# initialize algorithm with appointed hyperparameters
mappo = marl.algos.mappo(hyperparam_source='mpe')
print(type(mappo))

# build agent model based on env + algorithms + user preference
# return: (model(deep learning model), model_config)
model = marl.build_model(env, mappo,
                         {"core_arch": "mlp", "encode_layer": "128-256"})

# 输出一下model的内容（测试用）

# start training
mappo.fit(
    env, model,
    stop={'timesteps_total': 1000000},
    share_policy='group',
    checkpoint_freq=100,
    num_workers=1,
    num_envs_per_worker=4,
    num_gpus=0,
    seed=0,
    evaluation_interval=10,
    evaluation_num_episodes=10,
    evaluation_config={"explore": False},
    callbacks=UWBCustomMetricsCallbacks
    # local_dir="/home/coolas-fly/MARLlib/nohup_results/ray_results",
)
