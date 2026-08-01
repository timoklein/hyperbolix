"""Guard-rail tests for the ``hyperbolix`` root package surface (audit A11, E2.2).

Two properties the rest of the suite cannot see, because ``tests/conftest.py`` imports
``hyperbolix`` and enables ``jax_enable_x64`` before any test module is collected:

1. A *fresh* interpreter that does nothing but ``import hyperbolix`` must expose all six
   subpackages as attributes. Before A11 only three were imported at the root; the other
   three appeared as an attribute only after some *other* module happened to import them,
   so ``hyperbolix.distributions`` raised ``AttributeError`` from a clean start while the
   docs examples (which import the submodule first) worked.
2. Importing must not emit ``UserWarning``. ``uniform_poincare`` used to hold its
   Gauss-Legendre tables as module-level ``jnp.array(..., dtype=float64)``, which warns
   three times (and silently truncates to float32) whenever ``jax_enable_x64`` is off —
   i.e. on every plain ``import hyperbolix`` a user makes.

Both run in a subprocess: an in-process check would see the interpreter state that pytest's
own conftest has already established.
"""

from __future__ import annotations

import subprocess
import sys

import hyperbolix

# Every subpackage under ``hyperbolix/`` that ships an ``__init__.py``.
SUBPACKAGES = ("decomposition", "distributions", "manifolds", "nn_layers", "optim", "utils")

# The one non-subpackage name the root deliberately re-exports.
ROOT_SYMBOLS = ("LearnableCurvature",)


def _run_in_fresh_interpreter(code: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a clean interpreter (no pytest conftest, x64 off by default)."""
    return subprocess.run(
        [sys.executable, *extra_args, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_root_all_lists_exactly_the_six_subpackages_and_no_layers() -> None:
    """``__all__`` is the six subpackages plus ``LearnableCurvature`` — no stray layer re-exports.

    ``PoincareBatchNorm2D`` used to be re-exported here, alone among ~40 layers and with no
    rule explaining why. Re-adding any single layer at the root fails this test.
    """
    assert set(hyperbolix.__all__) == set(SUBPACKAGES) | set(ROOT_SYMBOLS)
    assert len(hyperbolix.__all__) == len(set(hyperbolix.__all__)), "duplicate entry in __all__"

    # Everything advertised is actually there.
    for name in hyperbolix.__all__:
        assert hasattr(hyperbolix, name), name

    # No layer leaked in via a bare ``from .nn_layers import X``.
    layer_names = set(hyperbolix.nn_layers.__all__)
    assert not (set(hyperbolix.__all__) & layer_names)
    assert not hasattr(hyperbolix, "PoincareBatchNorm2D")


def test_fresh_import_exposes_every_subpackage_as_attribute() -> None:
    """``import hyperbolix`` alone makes all six subpackages reachable by attribute."""
    probe = "; ".join(
        [
            "import hyperbolix",
            f"missing = [s for s in {SUBPACKAGES!r} if not hasattr(hyperbolix, s)]",
            "assert not missing, missing",
            "print('ok')",
        ]
    )
    result = _run_in_fresh_interpreter(probe)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "ok" in result.stdout


def test_fresh_import_emits_no_user_warning() -> None:
    """``import hyperbolix`` (and every subpackage it pulls in) is ``UserWarning``-free with x64 off."""
    result = _run_in_fresh_interpreter("import hyperbolix; print('ok')", "-W", "error::UserWarning")
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "ok" in result.stdout


def test_fresh_import_of_distributions_emits_no_user_warning() -> None:
    """The direct ``import hyperbolix.distributions`` path is warning-free too (audit E2.2)."""
    result = _run_in_fresh_interpreter("import hyperbolix.distributions; print('ok')", "-W", "error::UserWarning")
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "ok" in result.stdout
