from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, List

torch, nn = try_import_torch()


class MyCentralizedRelationalEncoder(nn.Module):
    """
    Relational critic encoder for UAV search task.

    Expected critic raw input:
        state: [B, num_agents, local_obs_dim + global_state_dim]

    Current fixed layout:
        local_obs_dim = 629
        global_state_dim = 165

    Shared global_state layout (taken from first agent slice only):
        1) agent_tokens_flat: [A * 5]
           each agent token:
             [pos_x, pos_y, vel_x, vel_y, dist_to_nearest_unfinished_target]

        2) target_tokens_flat: [T * 5]
           each target token:
             [target_x, target_y, found_flag, visible_flag, dist_to_nearest_agent]

        3) global_aux_flat: [100]
           visited_low
    """

    def __init__(self, model_config, obs_space):
        super(MyCentralizedRelationalEncoder, self).__init__()

        self.custom_config = model_config["custom_model_config"]
        self.activation = model_config.get("fcnet_activation")
        self.num_agents = self.custom_config["num_agents"]

        # -----------------------------
        # fixed dims (based on current env design)
        # -----------------------------
        self.local_obs_dim = 629
        self.global_state_dim = 165

        self.agent_token_dim = 5
        self.target_token_dim = 5
        self.num_targets = 10
        self.global_aux_dim = 100

        # global_state = agent_tokens_flat(3*5) + target_tokens_flat(10*5) + global_aux(100)
        self.agent_tokens_flat_dim = self.num_agents * self.agent_token_dim
        self.target_tokens_flat_dim = self.num_targets * self.target_token_dim

        assert self.agent_tokens_flat_dim + self.target_tokens_flat_dim + self.global_aux_dim == self.global_state_dim

        # -----------------------------
        # encoder dims
        # -----------------------------
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

        # hidden dims
        self.local_hidden_dim = max(final_out_dim // 2, 64)
        self.token_hidden_dim = 64
        self.relation_hidden_dim = 64
        self.global_aux_hidden_dim = 64

        # =========================================================
        # 1) Local team observation branch
        #    Input: all agents' local_obs => [B, A*629]
        # =========================================================
        local_input_dim = self.num_agents * self.local_obs_dim
        self.local_encoder = nn.Sequential(
            SlimFC(
                in_size=local_input_dim,
                out_size=self.local_hidden_dim,
                initializer=normc_initializer(1.0),
                activation_fn=self.activation,
            )
        )

        # =========================================================
        # 2) Agent token encoder (shared across agents)
        #    agent token dim = 5
        # =========================================================
        self.agent_token_encoder = nn.Sequential(
            SlimFC(
                in_size=self.agent_token_dim,
                out_size=self.token_hidden_dim,
                initializer=normc_initializer(1.0),
                activation_fn=self.activation,
            )
        )

        # =========================================================
        # 3) Target token encoder (shared across targets)
        #    target token dim = 5
        # =========================================================
        self.target_token_encoder = nn.Sequential(
            SlimFC(
                in_size=self.target_token_dim,
                out_size=self.token_hidden_dim,
                initializer=normc_initializer(1.0),
                activation_fn=self.activation,
            )
        )

        # =========================================================
        # 4) Agent-target relation encoder
        #    input: [agent_embed | target_embed] => 128
        # =========================================================
        self.relation_encoder = nn.Sequential(
            SlimFC(
                in_size=self.token_hidden_dim * 2,
                out_size=self.relation_hidden_dim,
                initializer=normc_initializer(1.0),
                activation_fn=self.activation,
            )
        )

        # =========================================================
        # 5) Global auxiliary encoder
        #    input: visited_low (100)
        # =========================================================
        self.global_aux_encoder = nn.Sequential(
            SlimFC(
                in_size=self.global_aux_dim,
                out_size=self.global_aux_hidden_dim,
                initializer=normc_initializer(1.0),
                activation_fn=self.activation,
            )
        )

        # =========================================================
        # 6) Final fusion
        #    [local_branch | relation_pool | global_aux]
        # =========================================================
        fusion_in_dim = self.local_hidden_dim + self.relation_hidden_dim + self.global_aux_hidden_dim
        self.encoder = nn.Sequential(
            SlimFC(
                in_size=fusion_in_dim,
                out_size=final_out_dim,
                initializer=normc_initializer(1.0),
                activation_fn=self.activation,
            )
        )

        self.output_dim = final_out_dim

    def forward(self, inputs) -> (TensorType, List[TensorType]):
        """
        inputs:
            [B, num_agents, local_obs_dim + global_state_dim]
            = [B, A, 629 + 165]
        """

        # ---------------------------------------------------------
        # 0) Split local_obs and shared global_state
        # ---------------------------------------------------------
        local_obs = inputs[:, :, :self.local_obs_dim]   # [B, A, 629]
        global_state = inputs[:, 0, self.local_obs_dim:self.local_obs_dim + self.global_state_dim]  # [B, 165]

        B = local_obs.shape[0]

        # ---------------------------------------------------------
        # 1) Local team observation branch
        # ---------------------------------------------------------
        local_obs_flat = local_obs.reshape(B, -1)  # [B, A*629]
        local_feat = self.local_encoder(local_obs_flat)  # [B, local_hidden_dim]

        # ---------------------------------------------------------
        # 2) Unpack structured global_state
        # ---------------------------------------------------------
        agent_tokens_flat = global_state[:, :self.agent_tokens_flat_dim]  # [B, 15]
        target_tokens_flat = global_state[
            :, self.agent_tokens_flat_dim:self.agent_tokens_flat_dim + self.target_tokens_flat_dim
        ]  # [B, 50]
        global_aux = global_state[:, self.agent_tokens_flat_dim + self.target_tokens_flat_dim:]  # [B, 100]

        agent_tokens = agent_tokens_flat.reshape(B, self.num_agents, self.agent_token_dim)  # [B, A, 5]
        target_tokens = target_tokens_flat.reshape(B, self.num_targets, self.target_token_dim)  # [B, T, 5]

        # ---------------------------------------------------------
        # 3) Encode agent / target tokens
        # ---------------------------------------------------------
        agent_embeds = self.agent_token_encoder(agent_tokens.reshape(B * self.num_agents, self.agent_token_dim))
        agent_embeds = agent_embeds.reshape(B, self.num_agents, self.token_hidden_dim)  # [B, A, 64]

        target_embeds = self.target_token_encoder(target_tokens.reshape(B * self.num_targets, self.target_token_dim))
        target_embeds = target_embeds.reshape(B, self.num_targets, self.token_hidden_dim)  # [B, T, 64]

        # ---------------------------------------------------------
        # 4) Build agent-target pair relations
        #    pair_ij = [agent_embed_i | target_embed_j]
        # ---------------------------------------------------------
        agent_expand = agent_embeds.unsqueeze(2).expand(B, self.num_agents, self.num_targets, self.token_hidden_dim)
        target_expand = target_embeds.unsqueeze(1).expand(B, self.num_agents, self.num_targets, self.token_hidden_dim)

        pair_feat = torch.cat([agent_expand, target_expand], dim=-1)  # [B, A, T, 128]
        pair_feat = pair_feat.reshape(B * self.num_agents * self.num_targets, self.token_hidden_dim * 2)

        relation_feat = self.relation_encoder(pair_feat)
        relation_feat = relation_feat.reshape(B, self.num_agents, self.num_targets, self.relation_hidden_dim)

        # ---------------------------------------------------------
        # 5) Relation pooling
        #    mean pool over all agent-target pairs
        # ---------------------------------------------------------
        relation_global = relation_feat.mean(dim=(1, 2))  # [B, relation_hidden_dim]

        # ---------------------------------------------------------
        # 6) Encode global auxiliary vector
        # ---------------------------------------------------------
        global_aux_feat = self.global_aux_encoder(global_aux)  # [B, global_aux_hidden_dim]

        # ---------------------------------------------------------
        # 7) Final fusion
        # ---------------------------------------------------------
        fused = torch.cat([local_feat, relation_global, global_aux_feat], dim=1)
        output = self.encoder(fused)

        # ---------------------------------------------------------
        # Debug print (only once)
        # ---------------------------------------------------------
        if not hasattr(self, "_debug_printed"):
            print("\n[MY CC ENCODER DEBUG]")
            print("inputs:", inputs.shape)
            print("local_obs:", local_obs.shape)
            print("global_state:", global_state.shape)
            print("agent_tokens:", agent_tokens.shape)
            print("target_tokens:", target_tokens.shape)
            print("global_aux:", global_aux.shape)
            print("local_feat:", local_feat.shape)
            print("agent_embeds:", agent_embeds.shape)
            print("target_embeds:", target_embeds.shape)
            print("relation_feat:", relation_feat.shape)
            print("relation_global:", relation_global.shape)
            print("global_aux_feat:", global_aux_feat.shape)
            print("fused:", fused.shape)
            print("output:", output.shape)
            print("\n[MY CC ENCODER DEBUG END]")

            self._debug_printed = True

        return output