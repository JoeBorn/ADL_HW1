from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm, BigNet

class Linear2Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size
        
        # Use 2-bit quantization for weights
        self.register_buffer(
            "weight_q2",
            torch.zeros(out_features * in_features // group_size, group_size // 4, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        
        # Optional bias in float32 (will be cast as needed)
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None

    def _quantize_2bit(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, self._group_size)
        norm = x.abs().max(dim=-1, keepdim=True).values
        x_norm = x / norm
        
        # Quantize to 2 bits (-1, -0.333, 0.333, 1)
        levels = torch.tensor([-1.0, -0.333, 0.333, 1.0], device=x.device)
        
        # Find closest level for each value
        distances = (x_norm.unsqueeze(-1) - levels.unsqueeze(0).unsqueeze(0))
        indices = distances.abs().argmin(dim=-1)
        x_q = torch.take(levels, indices)
        
        # Pack 4 2-bit values into one byte
        x_packed = torch.zeros(x.shape[0], x.shape[1] // 4, dtype=torch.int8, device=x.device)
        for i in range(4):
            x_packed = x_packed | (indices[:, i::4] << (i * 2))
        
        return x_packed, norm.to(torch.float16)

    def _dequantize_2bit(self, x_packed: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        # Unpack 2-bit values
        x_q = torch.zeros(x_packed.shape[0], x_packed.shape[1] * 4, device=x_packed.device)
        levels = torch.tensor([-1.0, -0.333, 0.333, 1.0], device=x_packed.device)
        
        for i in range(4):
            # Extract 2 bits and convert to long for indexing
            val = ((x_packed >> (i * 2)) & 0x3).long()
            x_q[:, i::4] = levels[val]
        
        return (x_q * norm.to(x_q.dtype)).view(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights and match input dtype
        weight = self._dequantize_2bit(self.weight_q2, self.weight_norm).to(x.dtype)
        weight = weight.reshape(self._shape)
        
        # Cast bias to match input dtype if it exists
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        
        return torch.nn.functional.linear(x, weight, bias)

class LowPrecisionBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear2Bit(channels, channels),
                torch.nn.ReLU(),
                Linear2Bit(channels, channels),
                torch.nn.ReLU(),
                Linear2Bit(channels, channels),
            )

        def forward(self, x):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
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

    def forward(self, x):
        return self.model(x)

def load(path: Path | None):
    # TODO (extra credit): Implement a BigNet that uses in
    # average less than 4 bits per parameter (<9MB)
    # Make sure the network retains some decent accuracy
    net = LowPrecisionBigNet()
    if path is not None:
        # Load the original model weights and quantize them
        original_net = BigNet()
        original_net.load_state_dict(torch.load(path, weights_only=True))
        
        # Copy and quantize weights
        with torch.no_grad():
            for (name, module), (_, orig_module) in zip(
                net.named_modules(), original_net.named_modules()
            ):
                if isinstance(module, Linear2Bit):
                    weight = orig_module.weight.data.flatten()
                    q2, norm = module._quantize_2bit(weight)
                    module.weight_q2.copy_(q2)
                    module.weight_norm.copy_(norm)
                    if module.bias is not None:
                        module.bias.data.copy_(orig_module.bias.data.to(torch.float16))
    
    return net
