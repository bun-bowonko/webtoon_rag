import torch
import torch.nn as nn
from peft.tuners.lora import Linear as LoRALinear


class LoRALinearExpertsWrapper(nn.Module):
    def __init__(self, moe_layer: nn.Module, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
        super().__init__()
        self.hidden_size = moe_layer.hidden_size
        self.expert_dim = moe_layer.expert_dim
        self.num_experts = moe_layer.num_experts
        self.act_fn = moe_layer.act_fn

        # 전문가별 LoRA 레이어를 ModuleList로 구성
        self.gate_up_proj = nn.ModuleList([
            LoRALinear(
                self.hidden_size, 2 * self.expert_dim,
                r=r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                fan_in_fan_out=False
            )
            for _ in range(self.num_experts)
        ])

        self.down_proj = nn.ModuleList([
            LoRALinear(
                self.expert_dim, self.hidden_size,
                r=r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                fan_in_fan_out=False
            )
            for _ in range(self.num_experts)
        ])

        # 기존 weight 복사
        with torch.no_grad():
            for i in range(self.num_experts):
                self.gate_up_proj[i].weight.copy_(
                    moe_layer.gate_up_proj[i].view(2 * self.expert_dim, self.hidden_size)
                )
                self.down_proj[i].weight.copy_(
                    moe_layer.down_proj[i].view(self.hidden_size, self.expert_dim)
                )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (num_experts, num_tokens_per_expert, hidden_size)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)

        outputs = []
        for i in range(self.num_experts):
            gate_up = self.gate_up_proj[i](hidden_states[i])  # (num_tokens, 2 * expert_dim)
            gate, up = gate_up.chunk(2, dim=-1)
            out = self.down_proj[i](up * self.act_fn(gate))    # (num_tokens, hidden_size)
            outputs.append(out)

        return torch.cat(outputs, dim=0)  # (total_tokens, hidden_size)
