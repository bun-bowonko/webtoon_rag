# model.py
import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from peft.tuners.lora.layer import LoRALinear  # 이미 정의된 LoRALinear 사용


class Llama4TextExpertsWithLoRA(nn.Module):
    def __init__(self, config, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        # 원래 weights
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))

        # LoRA 보조 경로
        self.lora_gate_up_proj = LoRALinear(self.hidden_size, 2 * self.expert_dim, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.lora_down_proj = LoRALinear(self.expert_dim, self.hidden_size, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # shape: (num_experts, tokens_per_expert, hidden_size)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)

        # base gate_up + LoRA
        gate_up = torch.bmm(hidden_states, self.gate_up_proj) + self.lora_gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)

        hidden = up * self.act_fn(gate)

        out = torch.bmm(hidden, self.down_proj) + self.lora_down_proj(hidden)

        return out.view(-1, self.hidden_size)
