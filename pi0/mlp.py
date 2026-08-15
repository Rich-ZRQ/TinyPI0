"""Gated feed-forward network used by Gemma."""

from torch import Tensor, nn
from torch.nn import functional as F

from configs.schema import TransformerConfig


class GemmaMLP(nn.Module):
    """Gemma gated MLP using GELU-tanh activation."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.width = config.width
        self.mlp_dim = config.mlp_dim

        self.gate_proj = nn.Linear(
            self.width,
            self.mlp_dim,
            bias=False,
        )
        self.up_proj = nn.Linear(
            self.width,
            self.mlp_dim,
            bias=False,
        )
        self.down_proj = nn.Linear(
            self.mlp_dim,
            self.width,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the Gemma gated feed-forward transformation."""

        self._validate_hidden_states(hidden_states)

        gate = F.gelu(
            self.gate_proj(hidden_states),
            approximate="tanh",
        )
        up = self.up_proj(hidden_states)

        return self.down_proj(gate * up)

    def _validate_hidden_states(
        self,
        hidden_states: Tensor,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must have shape [B, S, width], got {tuple(hidden_states.shape)}")

        if hidden_states.shape[-1] != self.width:
            raise ValueError(f"expected hidden width {self.width}, got {hidden_states.shape[-1]}")

        if not hidden_states.is_floating_point():
            raise TypeError(f"hidden_states must be floating point, got {hidden_states.dtype}")
