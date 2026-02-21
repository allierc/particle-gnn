# Refactoring Instructions — particle-gnn

refactor foolows previous refactor of NeuralGraph into flyvi-gnn
here similarly we refactor part of ParticleGrap into particle-gnn
the instruction for refactor written at first are in flyvis-gnn/refactor.md
current fil follows the refactor to particle-gnn

each step must be validated by running `python GNN_Test.py --config arbitrary --no-claude` (and `boids`, `gravity`). The test compares all key metrics against the reference baseline in `config/test_reference.json`.

use lower case in code writing, print and comments

**Status key**: [DONE] = merged, [PENDING] = not started, [PARTIAL] = in progress.

| Step | Description                                    | Status  |
| ---- | ---------------------------------------------- | ------- |
| 1    | Regression Test Infrastructure (`GNN_Test.py`) | DONE    |
| 2    | Rename `bSave` → `save`                        | DONE    |
| 3    | Eliminate Config Variable Unpacking            | DONE    |
| 4    | ParticleState Dataclass                        | PARTIAL |
| 5    | StrEnum Config Types                           | PENDING |
| 6    | FigureStyle + Plot Consolidation               | PENDING |
| 7    | Model Naming Cleanup                           | PENDING |
| 8    | LLM Exploration Readiness                      | PENDING |

---

## Step 1. Regression Test Infrastructure (`GNN_Test.py`) [DONE]

Already exists. Validates training metrics against `config/test_reference.json`.

```bash
python GNN_Test.py --config arbitrary --no-claude
python GNN_Test.py --config boids --no-claude
python GNN_Test.py --config gravity --no-claude
```

---

## Step 2. Rename `bSave` → `save` [PENDING]

### The problem

`bSave` is a camelCase outlier — same issue fixed in flyvis-gnn.

### The solution

Rename `bSave` → `save` in `graph_data_generator.py` and all callers.

### Files to modify

- `src/particle_gnn/generators/graph_data_generator.py`
- `GNN_Main.py`

### Validation

```bash
python GNN_Test.py --config arbitrary --no-claude
```

---

## Step 3. Eliminate Config Variable Unpacking [PENDING]

### The problem

`data_train_particle()` and `data_generate_particle()` each start with 20–30 lines of `variable = config.section.field` boilerplate.

### The solution

Three short section aliases (`sim`, `tc`, `mc`) then direct access: `sim.n_particles`, `tc.batch_size`. Only apply to variables that are NOT modified after unpacking.

### Files to modify

- `src/particle_gnn/models/graph_trainer.py` — `data_train_particle()`, `data_test()`
- `src/particle_gnn/generators/graph_data_generator.py` — `data_generate_particle()`

### Validation

```bash
python GNN_Test.py --config arbitrary --no-claude
```

---

## Step 4. ParticleState Dataclass [PENDING]

### The problem

Particle state is packed as `(N, 8)` for 2D / `(N, 10)` for 3D with dimension-dependent column indices:

| Field    | 2D index    | 3D index    |
| -------- | ----------- | ----------- |
| id       | `x[:, 0]`   | `x[:, 0]`   |
| position | `x[:, 1:3]` | `x[:, 1:4]` |
| velocity | `x[:, 3:5]` | `x[:, 4:7]` |
| type     | `x[:, 5]`   | `x[:, 7]`   |
| field    | `x[:, 6]`   | `x[:, 8]`   |
| age      | `x[:, 7]`   | `x[:, 9]`   |

Magic indices like `x[:, 1:dimension+1]`, `x[:, dimension+1:1+2*dimension]`, `x[:, 1+2*dimension]` appear 60+ times across 8 files.

### The solution

Create `src/particle_gnn/particle_state.py` with:

```python
@dataclass
class ParticleState:
    index: torch.Tensor        # (N,) long
    pos: torch.Tensor          # (N, dim) float32
    vel: torch.Tensor          # (N, dim) float32
    particle_type: torch.Tensor  # (N,) long
    field: torch.Tensor        # (N,) or (N, F) float32
    age: torch.Tensor          # (N,) float32

    @staticmethod
    def from_packed(x, dimension): ...
    def to_packed(self): ...
    def to(self, device): ...
    def clone(self): ...
```

### Migration strategy

Add `ParticleState` with `from_packed()`/`to_packed()`. Migrate one file at a time — use `to_packed()` at interfaces that still expect raw tensors. Once all consumers use ParticleState, remove packed paths.

### Files to modify

| File                                          | Changes                                       |
| --------------------------------------------- | --------------------------------------------- |
| `src/particle_gnn/particle_state.py`          | **Create** — dataclass                        |
| `generators/graph_data_generator.py`          | Use ParticleState for init and save           |
| `models/graph_trainer.py`                     | Convert to ParticleState at data load         |
| `models/Interaction_Particle.py`              | Access pos, vel, particle_type by name        |
| `models/Interaction_Particle_Field.py`        | Same                                          |
| `generators/PDE_A.py`, `PDE_B.py`, `PDE_G.py` | Use named fields                              |
| `utils.py`                                    | Update norm_velocity(), get_index_particles() |
| `models/utils.py`                             | Update plot/analysis functions                |

### Validation

```bash
python GNN_Test.py --config arbitrary --no-claude
python GNN_Test.py --config boids --no-claude
python GNN_Test.py --config gravity --no-claude
```

---

## Step 5. StrEnum Config Types [PENDING]

### The problem

`config.py` uses `Literal["periodic", "no", ...]` for 10+ fields. No IDE autocomplete.

### The solution

Add `StrEnum` classes (Python 3.10 compat shim) for: `Boundary`, `Prediction`, `Integration`, `UpdateType`, `GhostMethod`, `Sparsity`, `ClusterMethod`, `ClusterConnectivity`, `StateType`, `ConnectivityType`.

### Files to modify

- `src/particle_gnn/config.py`

### Validation

```bash
python GNN_Test.py --config arbitrary --no-claude
```

---

## Step 6. FigureStyle + Plot Consolidation [PENDING]

### The problem

Plot code scattered across 4 files. No centralized styling. Hardcoded font sizes (48, 32, 24).

### The solution

- Create `src/particle_gnn/figure_style.py` — `FigureStyle` dataclass with `default_style`, `dark_style`
- Create `src/particle_gnn/plot.py` — consolidate all visualization from `utils.py`, `models/utils.py`, `graph_trainer.py`, `graph_data_generator.py`
- All plot functions accept `style: FigureStyle = default_style`

### Files to modify

| File                                 | Changes               |
| ------------------------------------ | --------------------- |
| `src/particle_gnn/figure_style.py`   | **Create**            |
| `src/particle_gnn/plot.py`           | **Create**            |
| `utils.py`                           | Remove plot functions |
| `models/utils.py`                    | Remove plot functions |
| `models/graph_trainer.py`            | Extract inline plots  |
| `generators/graph_data_generator.py` | Import from plot.py   |

### Validation

```bash
python GNN_Test.py --config arbitrary --no-claude
```

---

## Step 7. Model Naming Cleanup [PENDING]

### The problem

`choose_training_model()` and `choose_model()` dispatch with if/elif chains, potential dead branches.

### The solution

Simplify dispatch, remove dead code paths.

### Files to modify

- `src/particle_gnn/models/utils.py`
- `src/particle_gnn/models/graph_trainer.py`

### Validation

```bash
python GNN_Test.py --config arbitrary --no-claude
```

---

## Step 8. LLM Exploration Readiness [PENDING]

### The problem

No `PARAMS_DOC` on models, no `LLM-MODIFIABLE` markers in trainer.

### The solution

- Add `PARAMS_DOC` to `Interaction_Particle` documenting equations, config params, typical ranges
- Add `LLM-MODIFIABLE` comment markers to `data_train_particle()` around optimizer setup, training loop, backward/step

### Files to modify

- `src/particle_gnn/models/Interaction_Particle.py`
- `src/particle_gnn/models/graph_trainer.py`

### Validation

```bash
python GNN_Test.py --config arbitrary --no-claude
```
