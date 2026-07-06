#!/usr/bin/env python3
"""Vendor a pinned, integrity-checked MathJax build into docs/javascripts/mathjax/.

Downloads the exact MathJax release from the npm registry, verifies its
SHA-512 integrity hash, and extracts the self-contained ``es5/`` bundle
(JS engine + CHTML web fonts) so the documentation loads MathJax from our
own domain with **zero third-party CDN at runtime**.

This exists because the docs previously loaded MathJax (and a now-hostile
``polyfill.io`` shim) from a CDN. Self-hosting removes the supply-chain and
CDN-trust surface entirely; pinning the hash makes the build tamper-evident.

Run before ``mkdocs build`` / ``mkdocs serve``:

    uv run python scripts/vendor_mathjax.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

MATHJAX_VERSION = "3.2.2"
# From: curl -s https://registry.npmjs.org/mathjax/3.2.2 \
#   | python3 -c "import json,sys; print(json.load(sys.stdin)['dist']['integrity'])"
# This is the part AFTER "sha512-". Bump it whenever MATHJAX_VERSION changes.
EXPECTED_SHA512_B64 = "Bt+SSVU8eBG27zChVewOicYs7Xsdt40qm4+UpHyX7k0/O9NliPc+x77k1/FEsPsjKPZGJvtRZM1vO+geW0OhGw=="

DEST = Path("docs/javascripts/mathjax")
PREFIX = "package/es5/"  # the self-contained engine + fonts live under es5/ in the npm tarball

# The docs load exactly one combined MathJax component — see mkdocs.yml
# `extra_javascript: javascripts/mathjax/tex-mml-chtml.js`. That component is
# self-contained JS (bundles loader + TeX/MathML input + CHTML output) and only
# lazy-loads CHTML web fonts from output/chtml/fonts/woff-v2/ at runtime.
#
# Vendoring the whole es5/ tree (~24 MB: SVG output, the speech-rule engine,
# every other component) ships ~22 MB per version that the page never requests.
# Because `mike` stores each docs version as a full independent copy on the
# gh-pages branch, that waste is multiplied by every release and pushed the
# published site (13 versions, 219 MB) past GitHub Pages' deploy limit, timing
# out `actions/deploy-pages` at its 10-minute ceiling. Vendor only what the page
# actually references (~1.5 MB/version).
KEEP_FILE = "tex-mml-chtml.js"  # the single component mkdocs loads
KEEP_DIR = "output/chtml/fonts/woff-v2/"  # CHTML web fonts it fetches on demand


def _wanted(name: str) -> bool:
    """True for the one component bundle and its CHTML web fonts (paths are es5-relative)."""
    return name == KEEP_FILE or name.startswith(KEEP_DIR)


def main() -> int:
    url = f"https://registry.npmjs.org/mathjax/-/mathjax-{MATHJAX_VERSION}.tgz"
    print(f"Fetching {url}")
    data = urllib.request.urlopen(url, timeout=60).read()  # noqa: S310 (https URL, hash-verified below)

    digest = base64.b64encode(hashlib.sha512(data).digest()).decode()
    if digest != EXPECTED_SHA512_B64:
        print(
            f"INTEGRITY MISMATCH — refusing to vendor:\n  expected sha512-{EXPECTED_SHA512_B64}\n  got      sha512-{digest}",
            file=sys.stderr,
        )
        return 1

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = []
        for member in tar.getmembers():
            if not member.name.startswith(PREFIX):
                continue
            member.name = member.name[len(PREFIX) :]  # strip "package/es5/"
            if member.name and _wanted(member.name):
                members.append(member)
        if not any(m.name == KEEP_FILE for m in members):
            print(f"EXPECTED FILE MISSING — {KEEP_FILE!r} not found in tarball es5/", file=sys.stderr)
            return 1
        # filter="data" (Python 3.12+) rejects path traversal / unsafe members.
        tar.extractall(DEST, members=members, filter="data")

    print(f"Vendored MathJax {MATHJAX_VERSION} -> {DEST} ({len(members)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
