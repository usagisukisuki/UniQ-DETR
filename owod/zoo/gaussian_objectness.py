"""C2: probabilistic objectness = Mahalanobis distance to a query-embedding Gaussian.

    d2(q) = (q - mu)^T Sigma^-1 (q - mu)
    o(q)  = exp(-tau * d2(q) / D)

The plan writes o(q) = exp(-1/2 d2). In D=256 dims d2 of an in-distribution sample is
~chi2(256) (mean 256), so exp(-d2/2) underflows to 0 for every query. We therefore use
the PROB scaling tau/D (PROB: tau=1.3), which keeps o(q) in a usable range.

mu / Sigma are buffers (saved in checkpoints, smoothed by ModelEMA) and never receive
gradients. The E-step pools matched embeddings over `interval` iterations and then
EMA-updates (mu, Sigma) with momentum m; Sigma gets shrinkage lambda*I.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

# cov_type='bn' is exactly PROB's ProbObjectnessHead: d2 = ||BatchNorm1d(q, affine=False)||^2 over all
# queries. In training the batch mean/var are used *with gradients*, so shrinking every embedding is
# normalised away and the matched-only loss can only move matched queries relative to the batch.
# With detached buffer statistics ('diag'/'full') the same loss collapses the embeddings and stalls
# classification (E2 pilot: closed-set Known AP50 0.07 vs 13.3 for E0 after 1 epoch).


class GaussianObjectness(nn.Module):
    def __init__(self, dim=256, momentum=0.1, shrinkage=1e-3, interval=50, temperature=1.3, cov_type='full'):
        super().__init__()
        assert cov_type in ('full', 'diag', 'bn')
        self.dim = dim
        self.momentum = momentum
        self.shrinkage = shrinkage
        self.interval = interval
        self.temperature = temperature
        self.cov_type = cov_type
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('cov', torch.eye(dim))
        self.register_buffer('initialized', torch.zeros((), dtype=torch.bool))
        self._reset_accum()
        self.num_fallbacks = 0
        if cov_type == 'bn':
            self.bn = nn.BatchNorm1d(dim, affine=False, momentum=momentum, eps=shrinkage)

    def _reset_accum(self):
        self._sum = None
        self._outer = None
        self._count = 0
        self._steps = 0

    # ---------------------------------------------------------------- inference
    def _whitener(self):
        """Returns a function x -> L^-1 x with Sigma = L L^T (or a diagonal fallback)."""
        cov = self.cov.float()
        if self.cov_type == 'full':
            L, info = torch.linalg.cholesky_ex(cov)
            if int(info) == 0:
                return lambda x: torch.linalg.solve_triangular(L, x.T, upper=False).T
            # fallback 1: eigen-decomposition with clamped spectrum (low-rank + floor)
            self.num_fallbacks += 1
            try:
                evals, evecs = torch.linalg.eigh(cov)
                inv_sqrt = evals.clamp(min=self.shrinkage).rsqrt()
                return lambda x: (x @ evecs) * inv_sqrt
            except RuntimeError:
                pass
        # fallback 2 / diag mode
        inv_std = cov.diagonal().clamp(min=self.shrinkage).rsqrt()
        return lambda x: x * inv_std

    def mahalanobis(self, x):
        """x: [..., D] -> squared Mahalanobis distance [...] (float32, grad flows into x only)."""
        shape = x.shape[:-1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.cov_type == 'bn':
                return self.bn(x.float().reshape(-1, self.dim)).pow(2).sum(-1).reshape(shape)
            z = x.float().reshape(-1, self.dim) - self.mean.float()
            w = self._whitener()(z)
            return w.pow(2).sum(-1).reshape(shape)

    def objectness(self, d2):
        return torch.exp(-self.temperature * d2 / self.dim)

    # ---------------------------------------------------------------- E-step
    @torch.no_grad()
    def accumulate(self, x):
        """x: [N, D] embeddings (all decoder queries or matched only, see OWCriterion.obj_stats).
        Call once per training iteration."""
        if self.cov_type == 'bn':
            return  # BatchNorm keeps its own running statistics
        x = x.detach().float().reshape(-1, self.dim)
        if self._sum is None:
            self._sum = torch.zeros(self.dim, device=x.device, dtype=torch.float64)
            self._outer = torch.zeros(self.dim, self.dim, device=x.device, dtype=torch.float64)
        self._sum += x.sum(0).double()
        self._outer += (x.T.double() @ x.double())
        self._count += x.shape[0]
        self._steps += 1
        # first estimate once enough samples for a well-conditioned covariance, then every `interval` iters
        if (not bool(self.initialized) and self._count >= 4 * self.dim) or self._steps >= self.interval:
            self._update()

    @torch.no_grad()
    def _update(self):
        s, o, n = self._sum, self._outer, torch.tensor(float(self._count), device=self._sum.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            for t in (s, o, n):
                dist.all_reduce(t)
        if n.item() < 2:
            self._reset_accum()
            return
        mean = s / n
        cov = (o - n * torch.outer(mean, mean)) / (n - 1)
        if self.cov_type == 'diag':
            cov = torch.diag(cov.diagonal())
        cov = cov + self.shrinkage * torch.eye(self.dim, device=cov.device, dtype=cov.dtype)
        if not bool(self.initialized):
            self.mean.copy_(mean.float())
            self.cov.copy_(cov.float())
            self.initialized.fill_(True)
        else:
            m = self.momentum
            self.mean.mul_(1 - m).add_(m * mean.float())
            # shrinkage is inside `cov`, so the EMA keeps the floor at lambda (plan's literal
            # `+ lambda*I` after the EMA would drift the floor to lambda/m)
            self.cov.mul_(1 - m).add_(m * cov.float())
        self._reset_accum()

    def extra_repr(self):
        return (f'dim={self.dim}, cov_type={self.cov_type}, momentum={self.momentum}, '
                f'shrinkage={self.shrinkage}, interval={self.interval}, temperature={self.temperature}')
