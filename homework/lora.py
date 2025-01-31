from pathlib import Path
import math

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        # Keep the LoRA layers in float32
        # Initialize the base half-precision linear layer
        super().__init__(in_features, out_features, bias)
        
        # Initialize LoRA layers in float32
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False)
        
        # Initialize weights - using kaiming_uniform for better training
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)
        
        # Make sure LoRA layers are trainable (float32)
        self.lora_a.weight.requires_grad_(True)
        self.lora_b.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward. Make sure to cast inputs to self.linear_dtype and the output back to x.dtype

        # Store original dtype
        original_dtype = x.dtype
        
        # Main path: half-precision linear layer
        main_output = super().forward(x)
        
        # LoRA path: keep in float32
        x_float32 = x.to(torch.float32)
        lora_output = self.lora_b(self.lora_a(x_float32))
        
        # Combine paths and convert back to original dtype
        return (main_output + lora_output).to(original_dtype)


class LoraBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision, and all linear layers have LoRA adapters.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()

            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)

        
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),  # Keep LayerNorm in full precision
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet(lora_dim=16)
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
