"""The fixed CrossPhase encoder.

This module is part of the frozen scaffold: the architecture, the initialisation
scheme and the parameter count are identical in every setting and for every
method.  Global average pooling lets the same network consume the different
sequence lengths used by the three settings without any per-setting branching.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossPhaseNet(nn.Module):
    """~9k-parameter 1-D CNN mapping a 2-channel signal to a single logit."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv1d(2, w, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(w, 2 * w, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(2 * w, 2 * w, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Linear(2 * w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.mean(dim=-1)
        return self.head(h).squeeze(-1)


def build_model(seed: int, width: int = 16) -> CrossPhaseNet:
    """Deterministic construction -- the only source of init randomness."""
    generator = torch.Generator().manual_seed(int(seed))
    model = CrossPhaseNet(width=width)
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                fan_in = param[0].numel()
                bound = (1.0 / fan_in) ** 0.5
                param.copy_(
                    torch.empty_like(param).uniform_(-bound, bound, generator=generator)
                )
            else:
                param.zero_()
    return model


def parameter_count(width: int = 16) -> int:
    return sum(p.numel() for p in CrossPhaseNet(width=width).parameters())
