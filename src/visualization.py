from marllib import marl
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# prepare the environment
env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="mpe")

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# rendering
mappo.render(env, model,
             restore_path={'params_path': "exp_results/mappo_mlp_simple_spread/MAPPOTrainer_mpe_simple_spread_9c90d_00000_0_2026-01-21_20-49-23/params.json",  # experiment configuration
                           'model_path': "exp_results/mappo_mlp_simple_spread/MAPPOTrainer_mpe_simple_spread_9c90d_00000_0_2026-01-21_20-49-23/checkpoint_000032/checkpoint-32",  # checkpoint path
                           'render': True},  # render
             local_mode=True,
             share_policy="all",
             checkpoint_end=False,
             num_gpus=0,
             num_gpus_per_worker=0
             )
