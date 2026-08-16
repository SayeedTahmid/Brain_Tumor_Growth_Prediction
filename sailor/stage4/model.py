"""A small 3D residual U-Net for the conditioning ladder.

AMD-007 fixes ONE architecture across C0-C4 and P1-P3. Rungs differ only in
which conditioning variables are supplied, never in capacity, depth or width —
otherwise a rung gap measures architecture rather than conditioning.

Conditioning enters through FiLM-style scale/shift on the bottleneck, so C0
(none) and C4 (all) are the SAME network with a conditioning vector that is
zero-length or populated. That is what makes the rungs comparable: no layer is
added or removed between them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

BASE_CHANNELS = 16
DEPTH = 3


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.a = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.na = nn.InstanceNorm3d(cout, affine=True)
        self.b = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.nb = nn.InstanceNorm3d(cout, affine=True)
        self.skip = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        h = self.act(self.na(self.a(x)))
        h = self.nb(self.b(h))
        return self.act(h + self.skip(x))


class ResidualUNet3D(nn.Module):
    """Predicts the target mask logits from the input mask (+ conditioning)."""

    def __init__(self, in_ch: int = 1, cond_dim: int = 0,
                 base: int = BASE_CHANNELS, depth: int = DEPTH):
        super().__init__()
        self.cond_dim = cond_dim
        chs = [base * (2 ** i) for i in range(depth + 1)]
        self.down = nn.ModuleList()
        c = in_ch
        for ch in chs[:-1]:
            self.down.append(Block(c, ch)); c = ch
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = Block(c, chs[-1])
        # Conditioning modulates the bottleneck only. With cond_dim = 0 the
        # film layer is absent and the forward path is byte-identical to C0.
        self.film = (nn.Linear(cond_dim, chs[-1] * 2) if cond_dim > 0 else None)
        self.up = nn.ModuleList()
        self.upconv = nn.ModuleList()
        rev = list(reversed(chs[:-1]))
        c = chs[-1]
        for ch in rev:
            self.upconv.append(nn.ConvTranspose3d(c, ch, 2, stride=2))
            self.up.append(Block(ch * 2, ch)); c = ch
        self.head = nn.Conv3d(c, 1, 1)

    def forward(self, x, cond=None):
        skips = []
        for blk in self.down:
            x = blk(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        if self.film is not None and cond is not None:
            gb = self.film(cond)
            g, b = gb.chunk(2, dim=1)
            x = x * (1 + g[..., None, None, None]) + b[..., None, None, None]
        for upc, blk, s in zip(self.upconv, self.up, reversed(skips)):
            x = upc(x)
            x = blk(torch.cat([x, s], dim=1))
        return self.head(x)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
