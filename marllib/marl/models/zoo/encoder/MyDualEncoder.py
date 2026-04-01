from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, List

torch, nn = try_import_torch()


class MyDualEncoder(nn.Module):
    """
    Dual-branch encoder for UAV task.

    actor_obs layout (NO global state):
        [ pos_norm(2) | vel_norm(2) | region_low(400) | local_patch(225) ]
        total = 629

    branch design:
        - coarse branch: pos + vel + region_low (404)
        - fine branch: local_patch (225)

    fusion design:
        - gated fusion between coarse and fine features
        - gate is learned from both branch features
    """

    def __init__(self, model_config, obs_space):
        super(MyDualEncoder, self).__init__()

        # -------- config --------
        self.custom_config = model_config["custom_model_config"]
        self.activation = model_config.get("fcnet_activation")

        # -------- dims --------
        self.actor_obs_dim = 629
        self.coarse_dim = 404
        self.fine_dim = 225

        # -------- encoder dims --------
        if "encode_layer" in self.custom_config["model_arch_args"]:
            encode_layer = self.custom_config["model_arch_args"]["encode_layer"]
            encoder_layer_dim = [int(i) for i in encode_layer.split("-")]
        else:
            encoder_layer_dim = []
            for i in range(self.custom_config["model_arch_args"]["fc_layer"]):
                out_dim = self.custom_config["model_arch_args"][f"out_dim_fc_{i}"]
                encoder_layer_dim.append(out_dim)

        self.encoder_layer_dim = encoder_layer_dim
        final_out_dim = self.encoder_layer_dim[-1]

        # 分支隐藏维度
        branch_hidden_dim = max(final_out_dim // 2, 64)

        # =========================================================
        # 1. Coarse branch（粗搜索）
        # =========================================================
        coarse_layers = [
            SlimFC(
                in_size=self.coarse_dim,
                out_size=branch_hidden_dim,
                initializer=normc_initializer(1.0),
                activation_fn=self.activation,
            )
        ]

        if len(self.encoder_layer_dim) > 1:
            coarse_layers.append(
                SlimFC(
                    in_size=branch_hidden_dim,
                    out_size=branch_hidden_dim,
                    initializer=normc_initializer(1.0),
                    activation_fn=self.activation,
                )
            )

        self.coarse_encoder = nn.Sequential(*coarse_layers)

        # =========================================================
        # 2. Fine branch（精细搜索，CNN over local_patch 15x15）
        # =========================================================
        self.patch_side = 15  # local_patch = 15 x 15 = 225

        self.fine_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 15 -> 7
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 7 -> 3
        )

        self.fine_fc = SlimFC(
            in_size=32 * 3 * 3,
            out_size=branch_hidden_dim,
            initializer=normc_initializer(1.0),
            activation_fn=self.activation,
        )

        # =========================================================
        # 3. Gated Fusion
        # =========================================================
        self.gate_layer = nn.Sequential(
            SlimFC(
                in_size=branch_hidden_dim * 2,
                out_size=branch_hidden_dim,
                initializer=normc_initializer(1.0),
                activation_fn=self.activation,
            ),
            SlimFC(
                in_size=branch_hidden_dim,
                out_size=branch_hidden_dim,
                initializer=normc_initializer(1.0),
                activation_fn=None,
            ),
        )

        self.encoder = nn.Sequential(
            SlimFC(
                in_size=branch_hidden_dim,
                out_size=final_out_dim,
                initializer=normc_initializer(1.0),
                activation_fn=self.activation,
            )
        )

        self.output_dim = final_out_dim

    def forward(self, inputs) -> (TensorType, List[TensorType]):
        """
        inputs shape:
            [B, obs_dim]

        obs_dim = actor_obs + global_state

        我们只取 actor_obs（前629）
        """

        inputs = inputs.reshape(inputs.shape[0], -1)

        # -------- 只取 actor obs --------
        actor_obs = inputs[:, :self.actor_obs_dim]

        # -------- 分支切分 --------
        coarse_input = actor_obs[:, :self.coarse_dim]  # [B, 404]
        fine_input = actor_obs[:, self.coarse_dim:self.actor_obs_dim]  # [B, 225]

        # -------- 编码 --------
        coarse_feat = self.coarse_encoder(coarse_input)

        # local_patch: [B, 225] -> [B, 1, 15, 15]
        fine_patch = fine_input.reshape(-1, 1, self.patch_side, self.patch_side)
        fine_conv_feat = self.fine_conv(fine_patch)
        fine_conv_feat = fine_conv_feat.reshape(fine_conv_feat.shape[0], -1)
        fine_feat = self.fine_fc(fine_conv_feat)

        # -------- Gated 融合 --------
        gate_input = torch.cat([coarse_feat, fine_feat], dim=1)
        gate = self.gate_layer(gate_input)
        gate = torch.sigmoid(gate)
        fused = gate * coarse_feat + (1.0 - gate) * fine_feat
        output = self.encoder(fused)

        # -------- debug（只打印一次）--------
        if not hasattr(self, "_debug_printed"):
            print("\n[MY DUAL ENCODER DEBUG]")
            print("actor_obs:", actor_obs.shape)
            print("coarse_input:", coarse_input.shape)
            print("fine_input:", fine_input.shape)
            print("fine_patch:", fine_patch.shape)
            print("fine_conv_feat:", fine_conv_feat.shape)
            print("coarse_feat:", coarse_feat.shape)
            print("fine_feat:", fine_feat.shape)
            print("gate_input:", gate_input.shape)
            print("gate:", gate.shape)
            print("fused:", fused.shape)
            print("output:", output.shape)
            print("[MY DUAL ENCODER DEBUG END]")
            self._debug_printed = True

        return output