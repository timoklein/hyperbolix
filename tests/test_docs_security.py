"""Guard-rail test: the external-host audit must flag disallowed resource hosts
(e.g. a re-introduced polyfill.io CDN) while ignoring plain anchor links.

Loads scripts/check_external_hosts.py by path since scripts/ is not a package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_checker() -> ModuleType:
    path = Path(__file__).parent.parent / "scripts" / "check_external_hosts.py"
    spec = importlib.util.spec_from_file_location("check_external_hosts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flags_external_script_but_not_anchor_links(tmp_path: Path) -> None:
    checker = _load_checker()
    (tmp_path / "index.html").write_text(
        '<script src="https://polyfill.io/v3/polyfill.min.js"></script>'  # disallowed resource
        '<script src="https://fonts.googleapis.com/x.js"></script>'  # allowlisted resource
        '<a href="https://arxiv.org/abs/1805.09112">citation</a>'  # link -> must be ignored
    )
    violations = checker.find_violations(tmp_path, checker.ALLOWED_HOSTS)
    assert violations == {"index.html": {"polyfill.io"}}


def test_clean_site_passes(tmp_path: Path) -> None:
    checker = _load_checker()
    (tmp_path / "page.html").write_text(
        '<script src="javascripts/mathjax/tex-mml-chtml.js"></script>'  # relative -> same origin
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css">'  # allowlisted
        '<link rel="canonical" href="https://timoklein.github.io/hyperbolix/">'  # self host
    )
    assert checker.find_violations(tmp_path, checker.ALLOWED_HOSTS) == {}
