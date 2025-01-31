from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Implement a half-precision Linear Layer.
        Feel free to use the torch.nn.Linear class as a parent class (it makes load_state_dict easier, names match).
        Feel free to set self.requires_grad_ to False, we will not backpropagate through this layer.
        """
        # Initialize the parent class (torch.nn.Linear)
        super().__init__(in_features, out_features, bias)
        
        # Convert weights to half precision
        self.weight.data = self.weight.data.to(torch.float16)
        
        # Convert bias to half precision if it exists
        if self.bias is not None:
            self.bias.data = self.bias.data.to(torch.float16)
        
        # Disable gradient computation since we won't backpropagate
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hint: Use the .to method to cast a tensor to a different dtype (i.e. torch.float16 or x.dtype)
        # The input and output should be of x.dtype = torch.float32
        
        # Store original dtype
        original_dtype = x.dtype
        
        # Convert input to half precision
        x_half = x.to(torch.float16)
        
        # Perform linear operation in half precision
        output = torch.nn.functional.linear(x_half, self.weight, self.bias)
        
        # Convert back to original dtype
        return output.to(original_dtype)


class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision. Make sure that the normalization uses full
    precision though to avoid numerical instability.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),  # Keep LayerNorm in full precision
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
