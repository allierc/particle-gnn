"""particle state and field state dataclasses for particle-gnn simulation.

replaces the packed (N, 8) [2D] or (N, 10) [3D] tensor with named fields.
the dimension-dependent column layout is encapsulated in from_packed() / to_packed().

packed tensor layout:
  column 0:             index (particle ID)
  columns 1:1+dim:      pos (position, 2D or 3D)
  columns 1+dim:1+2*dim: vel (velocity, 2D or 3D)
  column 1+2*dim:       particle_type
  column 2+2*dim:       field (H1, variable width)

classes:
  ParticleState       — single-frame particle state (N, C) -> named fields
  ParticleTimeSeries  — full simulation timeseries, static metadata + dynamic per-frame data
  FieldState          — single-frame mesh/field node state (same packed layout as particles)
  FieldTimeSeries     — full simulation timeseries for mesh/field nodes
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


# field classification for timeseries classes
# unlike flyvis-gnn (neurons don't move), particles move so pos is dynamic
STATIC_FIELDS = {'index', 'particle_type'}
DYNAMIC_FIELDS = {'pos', 'vel', 'field'}
ALL_FIELDS = STATIC_FIELDS | DYNAMIC_FIELDS

FIELD_STATIC_FIELDS = {'index', 'pos', 'particle_type'}
FIELD_DYNAMIC_FIELDS = {'vel', 'field'}
FIELD_ALL_FIELDS = FIELD_STATIC_FIELDS | FIELD_DYNAMIC_FIELDS


def _apply(tensor, fn):
    """apply fn to tensor if not None, else return None."""
    return fn(tensor) if tensor is not None else None


def _unpack(x, dimension):
    """unpack a (N, C) tensor into named fields dict."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x).float()
    else:
        t = x.float() if x.dtype != torch.float32 else x

    p1 = 1                      # pos start
    p2 = 1 + dimension           # pos end = vel start
    v2 = 1 + 2 * dimension       # vel end = type index
    type_idx = v2
    field_start = type_idx + 1

    result = dict(
        index=t[:, 0].long(),
        pos=t[:, p1:p2],
        vel=t[:, p2:v2],
        particle_type=t[:, type_idx].long(),
        field=None,
    )

    remaining = t.shape[1] - field_start
    if remaining >= 1:
        result['field'] = t[:, field_start:]

    return result


def _pack(index, pos, vel, particle_type, field, device):
    """pack named fields back into (N, C) tensor."""
    dim = pos.shape[1]
    n = pos.shape[0]
    n_cols = 1 + dim + dim + 1  # index + pos + vel + type
    if field is not None:
        field_cols = field.shape[1] if field.dim() > 1 else 1
        n_cols += field_cols

    x = torch.zeros(n, n_cols, dtype=torch.float32, device=device)
    col = 0

    if index is not None:
        x[:, col] = index.float()
    col += 1

    x[:, col:col + dim] = pos
    col += dim

    if vel is not None:
        x[:, col:col + dim] = vel
    col += dim

    if particle_type is not None:
        x[:, col] = particle_type.float()
    col += 1

    if field is not None:
        if field.dim() > 1:
            fc = field.shape[1]
            x[:, col:col + fc] = field
            col += fc
        else:
            x[:, col] = field
            col += 1

    return x


@dataclass
class ParticleState:
    """single-frame particle state for N particles.

    all fields default to None — only populated fields are used.
    """

    index: torch.Tensor | None = None           # (N,) long — particle IDs
    pos: torch.Tensor | None = None             # (N, dim) float32 — position
    vel: torch.Tensor | None = None             # (N, dim) float32 — velocity
    particle_type: torch.Tensor | None = None   # (N,) long — type label
    field: torch.Tensor | None = None           # (N, F) float32 — field / features

    @property
    def n_particles(self) -> int:
        """infer N from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[0]
        raise ValueError("ParticleState has no populated fields")

    @property
    def dimension(self) -> int:
        """infer spatial dimension from pos shape."""
        if self.pos is not None:
            return self.pos.shape[1]
        if self.vel is not None:
            return self.vel.shape[1]
        raise ValueError("cannot infer dimension: pos and vel are both None")

    @property
    def device(self) -> torch.device:
        """infer device from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("ParticleState has no populated fields")

    @classmethod
    def from_packed(cls, x: torch.Tensor | np.ndarray, dimension: int) -> ParticleState:
        """create from packed (N, C) tensor.

        2D layout (C=7+F): [index, x, y, vx, vy, type, field...]
        3D layout (C=8+F): [index, x, y, z, vx, vy, vz, type, field...]
        """
        d = _unpack(x, dimension)
        return cls(**d)

    def to_packed(self) -> torch.Tensor:
        """pack back into (N, C) tensor for legacy compatibility."""
        return _pack(self.index, self.pos, self.vel, self.particle_type,
                     self.field, self.device)

    def to(self, device: torch.device) -> ParticleState:
        """move all non-None tensors to device."""
        return ParticleState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.to(device))
            for f in dc_fields(self)
        })

    def clone(self) -> ParticleState:
        """deep clone all non-None tensors."""
        return ParticleState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.clone())
            for f in dc_fields(self)
        })

    def detach(self) -> ParticleState:
        """detach all non-None tensors from computation graph."""
        return ParticleState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.detach())
            for f in dc_fields(self)
        })

    def subset(self, ids) -> ParticleState:
        """select a subset of particles by index."""
        return ParticleState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t[ids])
            for f in dc_fields(self)
        })

    @classmethod
    def zeros(cls, n_particles: int, dimension: int = 2,
              device: torch.device = None) -> ParticleState:
        """create zero-initialized ParticleState."""
        return cls(
            index=torch.arange(n_particles, dtype=torch.long, device=device),
            pos=torch.zeros(n_particles, dimension, dtype=torch.float32, device=device),
            vel=torch.zeros(n_particles, dimension, dtype=torch.float32, device=device),
            particle_type=torch.zeros(n_particles, dtype=torch.long, device=device),
            field=torch.zeros(n_particles, 1, dtype=torch.float32, device=device),
        )


@dataclass
class ParticleTimeSeries:
    """full simulation timeseries — static metadata + dynamic per-frame data.

    static fields are stored once (same for all frames): index, particle_type.
    dynamic fields have a leading time dimension (T, N, ...): pos, vel, field.

    follows the NeuronTimeSeries pattern from flyvis-gnn, but pos is dynamic
    because particles move (neurons don't).
    """

    # static (stored once)
    index: torch.Tensor | None = None           # (N,) long
    particle_type: torch.Tensor | None = None   # (N,) long

    # dynamic (per frame) — unlike flyvis-gnn, particles move so pos is dynamic
    pos: torch.Tensor | None = None             # (T, N, dim) float32
    vel: torch.Tensor | None = None             # (T, N, dim) float32
    field: torch.Tensor | None = None           # (T, N, F) float32

    @property
    def n_frames(self) -> int:
        """infer T from the first non-None dynamic field."""
        for name in DYNAMIC_FIELDS:
            val = getattr(self, name)
            if val is not None:
                return val.shape[0]
        raise ValueError("ParticleTimeSeries has no dynamic fields")

    @property
    def n_particles(self) -> int:
        """infer N from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[-2] if f.name in DYNAMIC_FIELDS else val.shape[0]
        raise ValueError("ParticleTimeSeries has no populated fields")

    @property
    def dimension(self) -> int:
        """infer spatial dimension from pos (T, N, dim)."""
        if self.pos is not None:
            return self.pos.shape[2]
        if self.vel is not None:
            return self.vel.shape[2]
        raise ValueError("cannot infer dimension")

    @property
    def device(self) -> torch.device:
        """infer device from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("ParticleTimeSeries has no populated fields")

    def frame(self, t: int) -> ParticleState:
        """extract single-frame ParticleState at time t.

        static fields are shared (not cloned).
        dynamic fields are cloned so the caller can modify them.
        """
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name in DYNAMIC_FIELDS:
                kwargs[f.name] = val[t].clone()
            else:
                kwargs[f.name] = val
        return ParticleState(**kwargs)

    def to(self, device: torch.device) -> ParticleTimeSeries:
        """move all non-None tensors to device."""
        return ParticleTimeSeries(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.to(device))
            for f in dc_fields(self)
        })

    def subset_particles(self, ids) -> ParticleTimeSeries:
        """select a subset of particles by index."""
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name in DYNAMIC_FIELDS:
                kwargs[f.name] = val[:, ids]
            else:
                kwargs[f.name] = val[ids]
        return ParticleTimeSeries(**kwargs)

    @classmethod
    def from_packed(cls, arr: torch.Tensor | np.ndarray, dimension: int) -> ParticleTimeSeries:
        """create from packed (T, N, C) tensor or numpy array.

        2D layout (C=7+F): [index, x, y, vx, vy, type, field...]
        3D layout (C=8+F): [index, x, y, z, vx, vy, vz, type, field...]
        """
        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(arr).float()
        else:
            t = arr.float() if arr.dtype != torch.float32 else arr

        p1 = 1
        p2 = 1 + dimension
        v2 = 1 + 2 * dimension
        type_idx = v2
        field_start = type_idx + 1

        result = cls(
            # static — take from first frame
            index=t[0, :, 0].long(),
            particle_type=t[0, :, type_idx].long(),
            # dynamic — all frames (particles move, unlike flyvis-gnn neurons)
            pos=t[:, :, p1:p2],
            vel=t[:, :, p2:v2],
        )

        remaining = t.shape[2] - field_start
        if remaining >= 1:
            result.field = t[:, :, field_start:]

        return result

    @classmethod
    def load(cls, path: str | Path, dimension: int) -> ParticleTimeSeries:
        """load from .npy file (legacy format)."""
        path = Path(path)
        if path.suffix == '.npy' or path.with_suffix('.npy').exists():
            npy_path = path if path.suffix == '.npy' else path.with_suffix('.npy')
            return cls.from_packed(np.load(npy_path), dimension)
        raise FileNotFoundError(f"no .npy found at {path}")


@dataclass
class FieldState:
    """single-frame mesh/field node state.

    same packed layout as ParticleState — used for mesh nodes in particle-field coupling.
    """

    index: torch.Tensor | None = None           # (N,) long — node IDs
    pos: torch.Tensor | None = None             # (N, dim) float32 — position
    vel: torch.Tensor | None = None             # (N, dim) float32 — velocity (often zero for mesh)
    particle_type: torch.Tensor | None = None   # (N,) long — type label
    field: torch.Tensor | None = None           # (N, F) float32 — field values

    @property
    def n_nodes(self) -> int:
        """infer N from the first non-None field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[0]
        raise ValueError("FieldState has no populated fields")

    @property
    def dimension(self) -> int:
        if self.pos is not None:
            return self.pos.shape[1]
        if self.vel is not None:
            return self.vel.shape[1]
        raise ValueError("cannot infer dimension")

    @property
    def device(self) -> torch.device:
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("FieldState has no populated fields")

    @classmethod
    def from_packed(cls, x: torch.Tensor | np.ndarray, dimension: int) -> FieldState:
        """create from packed (N, C) tensor. same layout as ParticleState."""
        d = _unpack(x, dimension)
        return cls(**d)

    def to_packed(self) -> torch.Tensor:
        """pack back into (N, C) tensor for legacy compatibility."""
        return _pack(self.index, self.pos, self.vel, self.particle_type,
                     self.field, self.device)

    def to(self, device: torch.device) -> FieldState:
        return FieldState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.to(device))
            for f in dc_fields(self)
        })

    def clone(self) -> FieldState:
        return FieldState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.clone())
            for f in dc_fields(self)
        })

    def detach(self) -> FieldState:
        return FieldState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.detach())
            for f in dc_fields(self)
        })

    def subset(self, ids) -> FieldState:
        return FieldState(**{
            f.name: _apply(getattr(self, f.name), lambda t: t[ids])
            for f in dc_fields(self)
        })


@dataclass
class FieldTimeSeries:
    """full simulation timeseries for mesh/field nodes.

    static fields stored once: index, pos, particle_type.
    dynamic fields per frame: vel, field.
    """

    # static
    index: torch.Tensor | None = None           # (N,)
    pos: torch.Tensor | None = None             # (N, dim)
    particle_type: torch.Tensor | None = None   # (N,)

    # dynamic
    vel: torch.Tensor | None = None             # (T, N, dim)
    field: torch.Tensor | None = None           # (T, N, F)

    @property
    def n_frames(self) -> int:
        for name in FIELD_DYNAMIC_FIELDS:
            val = getattr(self, name)
            if val is not None:
                return val.shape[0]
        raise ValueError("FieldTimeSeries has no dynamic fields")

    @property
    def n_nodes(self) -> int:
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.shape[-2] if f.name in FIELD_DYNAMIC_FIELDS else val.shape[0]
        raise ValueError("FieldTimeSeries has no populated fields")

    @property
    def dimension(self) -> int:
        if self.pos is not None:
            return self.pos.shape[1]
        if self.vel is not None:
            return self.vel.shape[2]
        raise ValueError("cannot infer dimension")

    @property
    def device(self) -> torch.device:
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is not None:
                return val.device
        raise ValueError("FieldTimeSeries has no populated fields")

    def frame(self, t: int) -> FieldState:
        """extract single-frame FieldState at time t."""
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name in FIELD_DYNAMIC_FIELDS:
                kwargs[f.name] = val[t].clone()
            else:
                kwargs[f.name] = val
        return FieldState(**kwargs)

    def to(self, device: torch.device) -> FieldTimeSeries:
        return FieldTimeSeries(**{
            f.name: _apply(getattr(self, f.name), lambda t: t.to(device))
            for f in dc_fields(self)
        })

    def subset_nodes(self, ids) -> FieldTimeSeries:
        kwargs = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if val is None:
                kwargs[f.name] = None
            elif f.name in FIELD_DYNAMIC_FIELDS:
                kwargs[f.name] = val[:, ids]
            else:
                kwargs[f.name] = val[ids]
        return FieldTimeSeries(**kwargs)

    @classmethod
    def from_packed_list(cls, frames: list[torch.Tensor], dimension: int) -> FieldTimeSeries:
        """create from list of (N, C) packed tensors (one per frame).

        this matches the legacy x_mesh_list format.
        """
        stacked = torch.stack(frames)  # (T, N, C)
        p1 = 1
        p2 = 1 + dimension
        v2 = 1 + 2 * dimension
        type_idx = v2
        field_start = type_idx + 1

        result = cls(
            index=stacked[0, :, 0].long(),
            pos=stacked[0, :, p1:p2],
            particle_type=stacked[0, :, type_idx].long(),
            vel=stacked[:, :, p2:v2],
        )

        remaining = stacked.shape[2] - field_start
        if remaining >= 1:
            result.field = stacked[:, :, field_start:]

        return result
