# dicty3d_periodic_overdamped_ABP_spring_pairforce.py
# Overdamped 3D periodic Dicty-like ABM (ABP-style):
# - Field: FFT diffusion + decay + agent secretion sources
# - Agents: overdamped position update (not v), ABP polarity p_i (unit vector)
#   * polarity aligns toward grad u via torque-like relaxation + rotational noise
# - Pair interactions: spring repulsion + smooth gated spring adhesion (+ optional dashpot removed since no v)
# - Internal controller: FHN + adaptor J + theta baseline (used only for secretion here)

from dataclasses import dataclass
import numpy as np

# ---------------------- Utilities (periodic) ----------------------

def wrap_positions(x, L):
    x %= L
    return x

def seed_min_spacing_periodic(N, L, rmin, rng, tries_cap=40000):
    pts = []
    rmin2 = rmin*rmin
    for _ in range(tries_cap):
        x = rng.random(3) * L
        ok = True
        for p in pts:
            d = x - p
            d -= L * np.round(d / L)
            if d.dot(d) < rmin2:
                ok = False
                break
        if ok:
            pts.append(x)
            if len(pts) == N:
                return np.array(pts)
    raise RuntimeError("Could not place all points with requested spacing on a periodic box.")

# ---------------------- Smooth functions ----------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def smooth_gate(r, r_on, delta):
    # ~1 for r << r_on and ~0 for r >> r_on
    return 1.0 / (1.0 + np.exp((r - r_on) / max(delta, 1e-12)))

# ---------------------- Parameters ----------------------

@dataclass
class Params:
    # domain/grid
    L: float = 2.0
    Nx: int = 96

    # field
    D: float = 0.5
    lam: float = 0.5
    kappa: float = 1.0        # secretion deposition scale

    # agents
    N: int = 100
    rc: float = 0.05          # particle radius (contact distance r0=2rc)

    # overdamped mobilities / noise
    sigma_pos: float = 0.02   # position noise amplitude: dx_noise = sigma_pos*sqrt(dt)*N(0,1)

    Fmax_rep: float = 20.0     # cap on repulsion force magnitude (tune to prevent instability while allowing some softness)
    # ---------------------- Pair interactions: WCA + spring adhesion ----------------------
    use_pair_forces: bool = True

    mu_t: float = 1/30.0  # mobility for pair-force-induced drift (tune to get reasonable speeds)
    k_rep: float = 50.0     # repulsion stiffness (tune)
    k_adh: float = 50.0
    eps_attach: float = 0.4 # r_adh_on = (1+eps_attach)*r0 is the distance where adhesion turns on
    delta_adh: float = 0.001

    # time
    dt: float = 0.002
    T: float = 10.0
    save_every: int = 5


class Dicty3DPeriodicOverdampedABP:
    def __init__(self, params: Params, seed: int = 0):
        self.p = params
        self.rng = np.random.default_rng(seed)

        self.dx = params.L / params.Nx
        shape = (params.Nx, params.Nx, params.Nx)
        self.c = np.zeros(shape, dtype=np.float64)

        # contact distance
        self.r0 = 2.0 * self.p.rc

        # adhesion on distance
        self.r_adh_on = (1.0 + self.p.eps_attach) * self.r0

        # positions
        rmin = 1.9 * self.r0
        try:
            self.x = seed_min_spacing_periodic(self.p.N, self.p.L, rmin=rmin, rng=self.rng)
        except RuntimeError:
            self.x = self.rng.random((self.p.N, 3)) * self.p.L

    def _min_image(self, d):
        return d - self.p.L * np.round(d / self.p.L)

    # ---------------------- spring adhesion forces ----------------------
    def cap_force(self, F, Fmax, eps=1e-12, smooth=True):
        # F: (N,3)
        mag = np.linalg.norm(F, axis=1, keepdims=True)
        if smooth:
            # smooth saturation: tanh
            scale = (Fmax * np.tanh(mag / Fmax)) / (mag + eps)
        else:
            # hard cap
            scale = np.minimum(1.0, Fmax / (mag + eps))
        return F * scale
    
    # ---------------------- WCA + spring adhesion forces ----------------------
    def compute_pair_forces(self):
        p = self.p
        N, L = p.N, p.L
        if not p.use_pair_forces:
            return np.zeros((N, 3), dtype=np.float64)

        rcut_rep = self.r0
        rcut = max(rcut_rep, self.r_adh_on)
        rcut2 = rcut * rcut

        # linked-cell bins
        nb = max(1, int(np.floor(L / rcut)))
        cell = L / nb
        idx = (np.floor(self.x / cell).astype(int)) % nb

        bins = {}
        for i in range(N):
            bins.setdefault((idx[i,0], idx[i,1], idx[i,2]), []).append(i)

        neigh = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
        F = np.zeros((N, 3), dtype=np.float64)

        kadh = p.k_adh
        r0 = self.r0
        r_on = self.r_adh_on
        delta = p.delta_adh

        for i in range(N):
            cx, cy, cz = idx[i,0], idx[i,1], idx[i,2]
            cand = []
            for ox, oy, oz in neigh:
                nx, ny, nz = (cx+ox) % nb, (cy+oy) % nb, (cz+oz) % nb
                cand.extend(bins.get((nx,ny,nz), []))
            if not cand:
                continue

            ja = np.array(cand, dtype=int)
            ja = ja[ja > i]
            if ja.size == 0:
                continue

            d = self._min_image(self.x[ja] - self.x[i])
            r2 = np.sum(d*d, axis=1)
            m = (r2 > 0.0) & (r2 < rcut2)
            if not np.any(m):
                continue

            ja = ja[m]
            d = d[m]
            r = np.sqrt(r2[m])
            rhat = d / (r[:, None] + 1e-12)

            # ----- Soft-core repulsion for r < r0 (bounded, no singularity) -----
            Frep = np.zeros_like(rhat)
            mrep = r < r0
            if np.any(mrep):
                rr = r[mrep]
                fmag = p.k_rep * (r0 - rr)          # linear in overlap, bounded by k_rep*r0
                Frep[mrep] = fmag[:, None] * rhat[mrep]

            # Spring adhesion (attractive toward r0), smoothly gated
            g_on  = sigmoid((r - r0) / delta)       # ~0 below r0, ~1 above r0
            g_off = smooth_gate(r, r_on, delta)     # ~1 below r_on, ~0 above r_on
            g = g_on * g_off
            Fadh_on_j = (-kadh * g * (r - r0))[:, None] * rhat

            # cap Frep
            # Frep = self.cap_force(Frep, p.Fmax_rep)
            Fij_on_j = Frep + Fadh_on_j
            # Fij_on_j = Frep

            F[i] -= np.sum(Fij_on_j, axis=0)
            np.add.at(F, ja, Fij_on_j)

        return F

    # ---------------------- one time step ----------------------
    def step(self):
        p = self.p; dt = p.dt

        # ---- Pair forces ----
        Fpair = self.compute_pair_forces()
        self.Fpair_last = Fpair          # <-- add this

        drift = p.mu_t * self.Fpair_last
        # print("drift max:", np.max(np.linalg.norm(drift, axis=1)), " mean:", np.mean(np.linalg.norm(drift, axis=1)))

        noise = p.sigma_pos * np.sqrt(dt) * self.rng.standard_normal(self.x.shape)

        self.x = self.x + dt * drift  + noise
        wrap_positions(self.x, p.L)

    # ---------------------- run ----------------------
    def run(self):
        steps = int(np.round(self.p.T / self.p.dt))
        save_every = max(1, self.p.save_every)

        X, FPAIR = [], []

        self.Fpair_last = self.compute_pair_forces()
        X.append(self.x.copy()); 
        FPAIR.append(np.zeros((self.p.N, 3), dtype=np.float64))  # t0 placeholder

        for n in range(steps):
            self.step()

            if ((n + 1) % save_every) == 0:
                X.append(self.x.copy())
                FPAIR.append(self.Fpair_last.copy())

        return (np.array(X), np.array(FPAIR))

# ---------------------- driver ----------------------

def simulate(params_dict=None, seed=0):
    params = Params(**(params_dict or {}))
    sim = Dicty3DPeriodicOverdampedABP(params, seed=seed)
    return (*sim.run(), params)

if __name__ == "__main__":
    cfg = dict(
        L=1, Nx=96,
        N=1000, rc=0.05, dt=0.005, T=40.0, save_every=1,  # reduced from 5 for smoother video

        # overdamped + noise
        mu_t=1/30.0,  # scaling in pair-force
        sigma_pos=0.02, # translation noise

        # pair forces
        use_pair_forces=True,
    )

    X, FPAIR, P = simulate(cfg, seed=7) # U, PDIR, A, B, JJ, TH, Z, SRC, P

    print("snapshots:", FPAIR.shape[0], " grid:", FPAIR.shape[1:], " agents:", X.shape[1])
    np.savez_compressed(
        "data_particle_N1000_T40_dt0005_L1_position_pairforce_k50_noise.npz", # _springAdh
        X=X, FPAIR=FPAIR, params=P.__dict__
    )
    print("Saved data_particle_N1000_T40_dt0005_L1_position_pairforce_k50_noise.npz")