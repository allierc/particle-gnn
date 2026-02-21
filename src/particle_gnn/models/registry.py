"""Model registry for particle-gnn.

Two registries:
  - Simulator registry: maps config names to physics simulator classes (PDE_A, PDE_B, PDE_G)
  - Model registry: maps config names to learnable GNN classes (Interaction_Particle, Interaction_Particle_Field)

Usage:
    @register_simulator("PDE_A", "PDE_ParticleField_A")
    class PDE_A(nn.Module):
        ...

    @register_model("PDE_A", "PDE_B", "PDE_G")
    class Interaction_Particle(nn.Module):
        ...

    sim_cls = get_simulator_class("PDE_A")
    model_cls = get_model_class("PDE_A")
"""

_SIMULATOR_REGISTRY: dict[str, type] = {}
_MODEL_REGISTRY: dict[str, type] = {}
_simulators_discovered = False
_models_discovered = False


def _discover_simulators():
    global _simulators_discovered
    if _simulators_discovered:
        return
    _simulators_discovered = True
    import particle_gnn.generators.PDE_A  # noqa: F401
    import particle_gnn.generators.PDE_B  # noqa: F401
    import particle_gnn.generators.PDE_G  # noqa: F401


def _discover_models():
    global _models_discovered
    if _models_discovered:
        return
    _models_discovered = True
    import particle_gnn.models.Interaction_Particle  # noqa: F401
    import particle_gnn.models.Interaction_Particle_Field  # noqa: F401


def register_simulator(*names: str):
    """Class decorator that registers a simulator under one or more config names."""
    def decorator(cls):
        for name in names:
            if name in _SIMULATOR_REGISTRY:
                raise ValueError(
                    f"Simulator name '{name}' already registered to {_SIMULATOR_REGISTRY[name].__name__}"
                )
            _SIMULATOR_REGISTRY[name] = cls
        return cls
    return decorator


def register_model(*names: str):
    """Class decorator that registers a learnable model under one or more config names."""
    def decorator(cls):
        for name in names:
            if name in _MODEL_REGISTRY:
                raise ValueError(
                    f"Model name '{name}' already registered to {_MODEL_REGISTRY[name].__name__}"
                )
            _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_simulator_class(name: str) -> type:
    """Look up simulator class by config name."""
    _discover_simulators()
    if name not in _SIMULATOR_REGISTRY:
        available = sorted(_SIMULATOR_REGISTRY.keys())
        raise KeyError(f"Unknown simulator '{name}'. Available: {available}")
    return _SIMULATOR_REGISTRY[name]


def get_model_class(name: str) -> type:
    """Look up learnable model class by config name."""
    _discover_models()
    if name not in _MODEL_REGISTRY:
        available = sorted(_MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name]


def list_simulators() -> list[str]:
    """Return sorted list of all registered simulator names."""
    _discover_simulators()
    return sorted(_SIMULATOR_REGISTRY.keys())


def list_models() -> list[str]:
    """Return sorted list of all registered model names."""
    _discover_models()
    return sorted(_MODEL_REGISTRY.keys())
