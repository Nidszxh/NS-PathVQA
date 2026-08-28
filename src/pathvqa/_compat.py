"""Compatibility shims for artifacts created before the ``src/pathvqa`` restructure.

Importing this package module registers legacy module aliases so that
``torch.load(..., weights_only=False)`` can unpickle checkpoints saved by the
pre-restructure code layout (e.g. classes embedded as ``utils.config.Config``).
"""

import sys
import types


def _install_legacy_module_aliases() -> None:
    """Map pre-restructure module paths onto the packaged modules.

    Old checkpoints pickle their config as ``utils.config.*``. Without the
    alias, unpickling raises ``ModuleNotFoundError: No module named 'utils'``.
    The alias points at the *same* module object, so isinstance checks and
    attribute access keep working for both old and new artifacts.
    """
    legacy_pkg = "utils"
    if legacy_pkg in sys.modules:
        return  # something else already owns the name; do not shadow it

    from pathvqa.utils import config as config_module

    pkg = types.ModuleType(legacy_pkg)
    pkg.__path__ = []  # mark as package so "utils.config" resolves
    pkg.config = config_module  # type: ignore[attr-defined]
    sys.modules[legacy_pkg] = pkg

    sys.modules["utils.config"] = config_module


_install_legacy_module_aliases()


def migrate_legacy_scene_head(
    state_dict: "dict[str, object]",
) -> "tuple[dict[str, object], list[str]]":
    """Drop pre-removal density-head parameters from a legacy model state dict.

    The nuclei-density head was removed because it was produced but never
    consumed by any loss. Checkpoints saved before that removal still carry the
    unused ``scene_parser.density_classifier.*`` parameters; strip them so those
    checkpoints keep loading with ``strict=True`` instead of failing on
    unexpected keys. Returns ``(filtered_state_dict, removed_keys)``.
    """
    prefix = "scene_parser.density_classifier."
    removed = [k for k in state_dict if k.startswith(prefix)]
    if not removed:
        return state_dict, []
    filtered = {k: v for k, v in state_dict.items() if not k.startswith(prefix)}
    return filtered, removed


def is_pre_dsl_checkpoint(*state_dicts: "dict[str, object]") -> bool:
    """Detect checkpoints trained before the differentiable DSL interpreter.

    The DSL interpreter (count-query routing wired into
    ``SymbolicFusionEngine``) was added after some production checkpoints were
    trained; their state dicts contain no ``dsl_interpreter.*`` parameters.
    Building the module for such a checkpoint fails ``strict=True`` loading,
    while random-initializing it would silently mis-evaluate count queries.
    Instead, loaders call this predicate and build the model *without* the
    interpreter, reproducing the architecture that produced the checkpoint
    (``SymbolicFusionEngine`` treats a missing interpreter as the pre-DSL path).

    Returns ``True`` only when at least one state dict was given and *none* of
    them carry DSL parameters. Callers should pass every state dict they will
    load (typically the model and EMA dicts, which are always saved together).
    """
    if not state_dicts:
        return False
    return not any(str(k).startswith("dsl_interpreter.") for sd in state_dicts for k in sd)
