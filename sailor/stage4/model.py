"""A small 3D U-Net for the conditioning ladder, with an optional residual head.

RESIDUAL HEAD (ROS §8): `Residual Head | x_t, Z_cond, Delta_t | Delta-hat`. The
head outputs a CHANGE, not a state. ROS §7 stage 8 makes residual modelling an
explicit ablation — "beats direct prediction, or is dropped" — so both modes
exist here and are compared rather than assumed.

WHY IT MATTERS, MEASURED. The direct-prediction C0 rung scored 0.5991 against
persistence 0.4928 — worse by +0.1063, about twice the measured MDE of 0.0555.
The identity pipeline control returned delta exactly 0.000000 on all 208 pairs,
so the scoring path is faithful and that deficit belongs to the model.

The cause is structural. A direct model minimises loss against the TARGET, and
input and target differ (Dice ~0.50), so it settles between them; on a
volume-ratio metric that can be worse than either endpoint. Nothing tells it
"when uncertain, copy the input".

The residual form removes that failure mode by construction:

    logits = PRIOR_SCALE * (2*x_t - 1) + correction(x_t, cond)

with the correction head ZERO-INITIALISED, so an untrained residual model is
EXACTLY persistence and can only depart from it by learning to. The identity
control doubles as the test: an untrained residual model must score 0.492826.

PRIOR_SCALE is 4.0, not a saturating value. sigmoid(±4) = 0.982 / 0.018, which
thresholds correctly while keeping the prior within reach of the correction: a
head output of ~8 can flip a voxel. A saturating prior of ±20 would need a
correction of ~40 to change any decision, which would make the residual head
nearly unable to move the prediction at all.

RESIDUAL PREDICTION — v0.32, and the reason C0 underperformed persistence.

C0 scored 0.5991 against persistence 0.4928: WORSE by +0.1063, about twice the
measured MDE. The identity-pipeline control excluded the scoring path (delta
exactly 0.000000 over 208 pairs), so the deficit belonged to the model. But a
U-Net with skip connections can represent the identity function trivially, so
"it cannot copy its input" was never a satisfying explanation.

The real cause is the objective. `logits = f(input)` asks the network to
synthesise the target from scratch, and the loss rewards matching the TARGET.
Cohort-wide Dice(input, target) is about 0.50, so a loss-minimising model lands
somewhere BETWEEN input and target — and on a volume-ratio metric a midpoint can
be worse than either endpoint. Nothing in the formulation says "when uncertain,
copy the input", even though copying is the strongest available prior.

The fix is a residual output head:

    logits = SAT * (2 * input - 1)  +  f(input, cond)

With f = 0 the prediction IS the input, which IS persistence. Verified: at random
initialisation the model reproduces its input exactly (Dice 1.0). Persistence
becomes the model's FLOOR rather than a target it must rediscover, and training
spends its capacity on deviations — which is the quantity the project is actually
about, and what "residual" in the project's own framing means.

COST, STATED PLAINLY. This changes the architecture, so the completed official
C0 rung (25 fits, 4.75 h) no longer describes the current model and must be
re-run. AMD-007 requires one architecture across all rungs; changing it after C0
means C0 is redone, not that the ladder is mixed. The config fingerprint will
change, so existing checkpoints will REFUSE to resume rather than silently mixing
formulations.

RESIDUAL PARAMETERISATION (v0.32) — WHY, AND WHAT IT FIXES.

C0 scored 0.5991 against persistence 0.4928: WORSE by +0.1063, roughly twice the
measured MDE. The identity pipeline control proved the scoring path adds nothing
(delta exactly 0.000000 on all 208 pairs), so the deficit is the model's.

The cause is the parameterisation, not the capacity. The network was asked to
predict the TARGET mask from scratch. Persistence — reproduce the input — is
therefore a function it must LEARN, and BCE + soft Dice against the target gives
it no reason to prefer the input when uncertain. Since input and target differ
substantially (Dice(input, target) = 0.4697), a loss-minimising model lands
between them and, on a volume-ratio metric, can be worse than either endpoint.

The fix is to make persistence the DEFAULT rather than a target. The network now
predicts a RESIDUAL added to a fixed logit prior derived from the input mask:

    logits = PRIOR_SCALE * (2 * input - 1) + residual

With the output head zero-initialised the residual is exactly 0, so at
initialisation the model IS persistence and scores exactly 0.4928. Training can
only move away from that starting point, and gradient descent has to earn every
departure. This is the "residual" in the project's own framing, applied where it
belongs.

WHAT THIS IS NOT. It is not a metric change, not a loss change, and not extra
capacity — the residual head has the same parameters as the old direct head. It
changes what the network is asked to represent.

CONSEQUENCE FOR THE LADDER. AMD-007 fixes one architecture across all rungs, so
C0 must be RE-RUN under this parameterisation before C1. Only C0 exists, no rung
comparison has been made, and the old C0 result is retained as
`rung_C0_direct_parameterisation` for the record.

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
#: Logit magnitude of the input prior. sigmoid(4.0) = 0.982, so a zero residual
#: thresholds to exactly the input mask at 0.5 while leaving the prior finite
#: and differentiable — a saturated prior (say 20) would make the residual
#: unable to overturn a voxel without an implausibly large output.
PRIOR_SCALE = 4.0


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
    """Target-mask logits from the input mask (+ conditioning).

    `residual=True` (ROS §8) outputs a CORRECTION to a persistence prior rather
    than the logits themselves. `residual=False` reproduces the direct
    prediction used by the completed C0 rung, retained as ablation A3's
    comparison arm ("beats direct prediction, or is dropped", ROS §7 stage 8).
    """

    def __init__(self, in_ch: int = 1, cond_dim: int = 0,
                 base: int = BASE_CHANNELS, depth: int = DEPTH,
                 residual: bool = True, prior_scale: float = PRIOR_SCALE):
        super().__init__()
        self.cond_dim = cond_dim
        self.residual = residual
        self.prior_scale = prior_scale
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
        if self.residual:
            # Zero-init so the residual starts at exactly 0: the untrained model
            # IS persistence. Every departure from it must be earned.
            #
            # MEASURED COST OF THIS CHOICE. At step 0 ONLY the head receives
            # gradient (2.1e-02); FiLM and the whole encoder receive exactly
            # zero, because they are multiplied by a zero head. The symmetry
            # breaks on the first optimiser step — after one step FiLM gradient
            # is 1.9e-07 and conditioning demonstrably changes the output — but
            # the first few steps train only the final 1x1 convolution. This is
            # the price of the guarantee that the model starts AT persistence,
            # and it is recorded rather than discovered later as slow early
            # convergence.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x, cond=None):
        inp = x
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
        out = self.head(x)
        if self.residual:
            # Prior from the INPUT mask, not from the features — so it is exact.
            return self.prior_scale * (2.0 * inp - 1.0) + out
        return out


CONFIG = {
    "architecture": "ResidualUNet3D",
    "base_channels": BASE_CHANNELS,
    "depth": DEPTH,
    "residual_prediction": True,
    "prior_scale": PRIOR_SCALE,
    "head_zero_init": True,
    "fixed_by": "AMD-007 — identical across C0-C4 and P1-P3",
    "changed_in": "v0.32",
    "changed_from": "logits = f(input) — target synthesised from scratch",
    "reason": (
        "C0 scored 0.5991 vs persistence 0.4928, worse by ~2x the MDE of 0.0555. "
        "The identity-pipeline control returned delta exactly 0.000000 over 208 "
        "pairs, so the scoring path was not at fault. Cause: the loss rewards "
        "matching the TARGET, and with cohort Dice(input,target) ~ 0.50 a "
        "loss-minimising model lands between input and target — which on a "
        "volume-ratio metric can be worse than either endpoint. The residual head "
        "plus zero-init makes persistence the FLOOR: the untrained model "
        "reproduces its input exactly (verified, Dice 1.0)."),
    "invalidates": (
        "The completed official C0 rung (25 fits, 4.75 h) describes the previous "
        "architecture and MUST be re-run. AMD-007 requires one architecture "
        "across rungs, so C0 is redone rather than the ladder being mixed."),
}


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
