from marllib import marl

# prepare env
env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# initialize algorithm with appointed hyperparameters
mappo = marl.algos.mappo(hyperparam_source='mpe')

# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start training
mappo.fit(env, model, stop={'timesteps_total': 100000}, share_policy='group', checkpoint_freq=100)