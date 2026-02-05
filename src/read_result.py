import json

result_path = "/home/coolas-fly/MARLlib/exp_results/mappo_mlp_Scenario1/MAPPOTrainer_uwb_planning_env_Scenario1_fa19c_00000_0_2026-02-05_19-44-06/result.json"  # 改成你的 result.json 路径

with open(result_path, "r") as f:
    last_line = None
    for line in f:
        last_line = line

data = json.loads(last_line)

for line in data:
    print(line, data[line])