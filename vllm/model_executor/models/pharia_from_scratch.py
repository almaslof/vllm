from torch import nn
import torch
from vllm.config import VllmConfig
from vllm.attention import Attention
from vllm.configuration_pharia import PhariaConfig


class LlamaAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scaling: float,
        config: PhariaConfig,
        prefix: str,
    ):
        super().__init__()
        self.attn = Attention(num_heads, head_size, scaling, prefix=f"{prefix}.attn")

from transformers.activations import ACT2FN
class PhariaMLP(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        o = self.down_proj(self.act_fn(self.up_proj(x)))
        return o

class PhariaDecoderLayer(nn.Module):
    def __init__(self, config: PhariaConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.self_attn = LlamaAttention(
            num_heads=2, head_size=32, scaling=1.0, config=config, prefix=f"{prefix}.self_attn"
        )
        #self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.input_layernorm = nn.LayerNorm(4608)
        self.mlp = PhariaMLP(config, layer_idx=layer_idx)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)


class PhariaModel(nn.Module):
    def __init__(self, 
                 *, 
                 vllm_config: VllmConfig, 
                 prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        #self.layers = nn.ModuleList(
        #    [
        #        PhariaDecoderLayer(vllm_config, layer_idx, prefix=f"{prefix}.layers.{i}")
        #        for i in range(vllm_config.model_config.hf_config.num_hidden_layers)
        #    ]
        #)

        self.layers = nn.ModuleList(
            [
                PhariaDecoderLayer(config, layer_idx, prefix=f"{prefix}.layers.{layer_idx}")
                for layer_idx in range(config.num_hidden_layers)
            ]
        )


class PhariaForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config #.model_config.hf_config
        self.config = config
        self.model = PhariaModel(vllm_config=config, prefix=f"{prefix}.model")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.get_input_embeddings(input_ids)
        return hidden_states
