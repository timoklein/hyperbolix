"""Import every module in the package, one test per module.

Guard against the bug class where a module references a name it never imports. Ruff's F821 is
off for ``hyperbolix/`` (jaxtyping shape strings, see ``pyproject.toml``) and pyright was
excluding the whole tree, so nothing was checking.

Scope, stated honestly: today every module is reachable from ``import hyperbolix``, so a
module-scope NameError already breaks collection everywhere. What this file adds is (a) a
failure that *names* the offending module instead of exploding inside ``conftest.py``, and (b)
coverage that survives the package's ``__init__`` chain changing -- a module that stops being
re-exported silently loses all import coverage otherwise. Undefined names inside *function*
bodies are not import-time failures at all; those are pyright's ``reportUndefinedVariable``.

Imports only: no JAX computation, so the whole file runs in well under a second.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

import pytest

import hyperbolix

# A package that fails to import cannot be walked, so ``walk_packages`` reports it through
# ``onerror`` and skips its children. Record those instead of letting them vanish from the
# parameter list -- a module that disappears is exactly the failure this file is meant to see.
_WALK_ERRORS: dict[str, BaseException | None] = {}


def _record_walk_error(name: str) -> None:
    _WALK_ERRORS[name] = sys.exc_info()[1]


_DISCOVERED = sorted(
    m.name for m in pkgutil.walk_packages(hyperbolix.__path__, prefix="hyperbolix.", onerror=_record_walk_error)
)
MODULES = ["hyperbolix", *sorted({*_DISCOVERED, *_WALK_ERRORS})]


def test_walk_found_the_package() -> None:
    """Guard the guard: an empty or near-empty walk would make every other test here vacuous."""
    assert len(MODULES) > 20, f"pkgutil.walk_packages found only {len(MODULES)} modules: {MODULES}"


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    """Importing the module raises nothing (missing imports, typos, circular imports)."""
    if module_name in _WALK_ERRORS:
        raise AssertionError(f"{module_name} failed to import during package discovery") from _WALK_ERRORS[module_name]
    importlib.import_module(module_name)
