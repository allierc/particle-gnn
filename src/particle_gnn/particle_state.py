"""Particle state dataclass for particle-gnn simulation.

Replaces the packed (N, 8) [2D] or (N, 10) [3D] tensor with named fields.
The dimension-dependent column layout is encapsulated in from_packed() / to_packed().

Packed tensor layout:
  Column 0:             index (particle ID)
  Columns 1:1+dim:      pos (position, 2D or 3D)
  Columns 1+dim:1+2*dim: vel (velocity, 2D or 3D)
  Column 1+2*dim:       particle_type
  Column 2+2*dim:       field (H1)
  Column 3+2*dim:       age (A1) [2D only, or additional field columns in 3D]
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields

import numpy as np
import torch


def _apply(tensor, fn):
    """Apply fn to tensor if not None, else return None."""
    return fn(tensor) if tensor is not None else None


@dataclass
class ParticleState:
    """Single-frame particle state for N particles.

    All fields default to None — only populated fields are used.
    """

    index: torch.Tensor | None = None           # (N,) long — particle IDs
    pos: torch.Tensor | None = None             # (N, dim) float32 — position
    vel: torch.Tensor | None = None             # (N, dim) float32 — velocity
    particle_type: torch.Tensor | None = None   # (N,) long — type label
    field: torch.Tensor | None = None           # (N, F) float32 — field / features
    age: torch.Tensor | None = None             # (N,) float32 — age / extra

    @property
    def n_particles(self) -> int:
        """Infer N from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[0]
        raise ValueError("ParticleState has no populated fields")

    @property
    def dimension(self) -> int:
        """Infer spatial dimension from pos shape."""
        if self.pos is not None:
            return self.pos.shape[1]
        if self.vel is not None:
            return self.vel.shape[1]
        raise ValueError("Cannot infer dimension: pos and vel are both None")

    @property
    def device(self) -> torch.device:
        """Infer device from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("ParticleState has no populated fields")

    @classmethod
    def from_packed(cls, x: torch.Tensor | np.ndarray, dimension: int) -> ParticleState:
        """Create from packed (N, C) tensor.

        2D layout (C=8): [index, x, y, vx, vy, type, field, age]
        3D layout (C=10): [index, x, y, z, vx, vy, vz, type, field1, field2]
        """
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
        else:
            t = x.float() if x.dtype != torch.float32 else x

        p1 = 1                      # pos start
        p2 = 1 + dimension           # pos end = vel start
        v2 = 1 + 2 * dimension       # vel end = type index
        type_idx = v2
        field_start = type_idx + 1

        result = cls(
            index=t[:, 0].long(),
            pos=t[:, p1:p2],
            vel=t[:, p2:v2],
            particle_type=t[:, type_idx].long(),
        )

        # Remaining columns are field/age
        remaining = t.shape[1] - field_start
        if remaining >= 2:
            result.field = t[:, field_start:field_start + 1]
            result.age = t[:, field_start + 1]
        elif remaining == 1:
            result.field = t[:, field_start:field_start + 1]

        return result

    def to_packed(self) -> torch.Tensor:
        """Pack back into (N, C) tensor for legacy compatibility.

        Returns (N, 8) for 2D or (N, 10) for 3D.
        """
        dim = self.dimension
        n = self.n_particles
        # Determine total columns
        n_cols = 1 + dim + dim + 1  # index + pos + vel + type
        if self.field is not None:
            field_cols = self.field.shape[1] if self.field.dim() > 1 else 1
            n_cols += field_cols
        if self.age is not None:
            n_cols += 1

        x = torch.zeros(n, n_cols, dtype=torch.float32, device=self.device)
        col = 0

        if self.index is not None:
            x[:, col] = self.index.float()
        col += 1

        if self.pos is not None:
            x[:, col:col + dim] = self.pos
        col += dim

        if self.vel is not None:
            x[:, col:col + dim] = self.vel
        col += dim

        if self.particle_type is not None:
            x[:, col] = self.particle_type.float()
        col += 1

        if self.field is not None:
            if self.field.dim() > 1:
                field_cols = self.field.shape[1]
                x[:, col:col + field_cols] = self.field
                col += field_cols
            else:
                x[:, col] = self.field
                col += 1

        if self.age is not None:
            x[:, col] = self.age
            col += 1

        return x

    def to(self, device: torch.device) -> ParticleState:
        """Move all non-None tensors to device."""
        return ParticleState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.to(device))
            for f in dc_fields(self)
        })

    def clone(self) -> ParticleState:
        """Deep clone all non-None tensors."""
        return ParticleState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.clone())
            for f in dc_fields(self)
        })

    def detach(self) -> ParticleState:
        """Detach all non-None tensors from computation graph."""
        return ParticleState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.detach())
            for f in dc_fields(self)
        })

    def subset(self, ids) -> ParticleState:
        """Select a subset of particles by index."""
        return ParticleState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t[ids])
            for f in dc_fields(self)
        })

    @classmethod
    def zeros(cls, n_particles: int, dimension: int = 2,
              device: torch.device = None) -> ParticleState:
        """Create zero-initialized ParticleState."""
        return cls(
            index=torch.arange(n_particles, dtype=torch.long, device=device),
            pos=torch.zeros(n_particles, dimension, dtype=torch.float32, device=device),
            vel=torch.zeros(n_particles, dimension, dtype=torch.float32, device=device),
            particle_type=torch.zeros(n_particles, dtype=torch.long, device=device),
            field=torch.zeros(n_particles, 1, dtype=torch.float32, device=device),
            age=torch.zeros(n_particles, dtype=torch.float32, device=device),
        )
