import torch
import torch.nn as nn
from peft.tuners.lora import Linear as LoRALinear


class LoRALinearExpertsWrapper(nn.Module):
    def __init__(self, moe_layer: nn.Module, r: int = 64, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__()
        self.hidden_size = moe_layer.hidden_size
        self.expert_dim = moe_layer.expert_dim
        self.num_experts = moe_layer.num_experts
        self.act_fn = moe_layer.act_fn

        # LoRA로 래핑된 선형 레이어 정의
        self.gate_up_proj = LoRALinear(
            self.hidden_size, 2 * self.expert_dim,
            r=r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=False
        )
        self.down_proj = LoRALinear(
            self.expert_dim, self.hidden_size,
            r=r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            fan_in_fan_out=False
        )

        # 기존 weight 복사
        with torch.no_grad():
            self.gate_up_proj.weight.copy_(moe_layer.gate_up_proj.view(-1, self.hidden_size))
            self.down_proj.weight.copy_(moe_layer.down_proj.view(-1, self.expert_dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        next_states = self.down_proj(up * self.act_fn(gate))
        return next_states.view(-1, self.hidden_size)
