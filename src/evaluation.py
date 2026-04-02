# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
example of how to render a trajectory on MPE
"""

from marllib import marl
from src.uwb_callback import UWBCustomMetricsCallbacks

# prepare the environment
env = marl.make_env(environment_name="uwb_planning_env", map_name="Scenario1")

# initialize algorithm and load hyperparameters
marl_algo = marl.algos.mappo(hyperparam_source='mpe')

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, marl_algo,
                         {"core_arch": "mlp", "encode_layer": "256-512"})

# rendering
marl_algo.render(env, model,
                 restore_path={
                     'params_path': "/home/coolas-fly/MARLlib/exp_results/mappo_mlp_Scenario1/New2/params.json",
                     # experiment configuration
                     'model_path': "/home/coolas-fly/MARLlib/exp_results/mappo_mlp_Scenario1/New2/checkpoint_000600/checkpoint-600",
                     # checkpoint path
                     'render': False},  # render
                 callbacks=UWBCustomMetricsCallbacks,
                 share_policy='group',
                 checkpoint_freq=10,
                 checkpoint_at_end=True,
                 num_workers=4,
                 num_envs_per_worker=2,
                 num_gpus=0,
                 seed=42,
                 checkpoint_end=False)
