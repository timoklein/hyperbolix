#!/usr/bin/env python3
"""Fail the docs build if the rendered site loads a resource from a host
outside the allowlist.

Guards against supply-chain CDN references (e.g. the ``polyfill.io`` incident)
silently re-entering the documentation. Only *resource loads* are enforced —
``<script src>``, ``<link href>``, ``<img src>``, and CSS ``url()`` / ``@import`` —
because those execute or style the page. Plain ``<a href>`` links (citations,
external references) are intentionally ignored, so prose never trips the guard.

Usage:
    python scripts/check_external_hosts.py site
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

# Third-party hosts the docs may load resources from at runtime.
# Keep this MINIMAL: every entry is a domain your readers' browsers must trust.
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "fonts.googleapis.com",  # Material theme web-font stylesheet
        "fonts.gstatic.com",  # Material theme web-font files
        # "cdn.jsdelivr.net",  # MathJax — keep ONLY if not self-hosting via vendor_mathjax.py
    }
)

# Your own GitHub Pages host(s): absolute self-links are not third parties.
SELF_HOSTS: frozenset[str] = frozenset({"timoklein.github.io", "hyperbolix.github.io"})

_HOST_RE = re.compile(r"https?://([^/\"'\s)]+)", re.IGNORECASE)
# CSS url(...) and @import targets, inside <style> blocks and .css files.
_CSS_URL_RE = re.compile(r"""(?:url\(|@import\s+)['"]?\s*(https?://[^/\"'\s)]+)""", re.IGNORECASE)


class _ResourceHostCollector(HTMLParser):
    """Collects external hosts from resource-loading tags only (not <a href>)."""

    _RESOURCE_ATTR = {
        "script": "src",
        "link": "href",
        "img": "src",
        "source": "src",
        "iframe": "src",
        "embed": "src",
        "object": "data",
    }

    def __init__(self) -> None:
        super().__init__()
        self.hosts: set[str] = set()
        self._in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "style":
            self._in_style = True
        attr = self._RESOURCE_ATTR.get(tag)
        if attr is None:
            return
        for name, value in attrs:
            if name == attr and value:
                match = _HOST_RE.match(value)
                if match:
                    self.hosts.add(match.group(1).lower())

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._in_style = False

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.hosts |= _hosts_in_css(data)


def _hosts_in_css(text: str) -> set[str]:
    hosts: set[str] = set()
    for match in _CSS_URL_RE.finditer(text):
        host = _HOST_RE.match(match.group(1))
        if host:
            hosts.add(host.group(1).lower())
    return hosts


def _hosts_in_html(text: str) -> set[str]:
    parser = _ResourceHostCollector()
    parser.feed(text)
    return parser.hosts


def find_violations(site_dir: Path, allowed: frozenset[str]) -> dict[str, set[str]]:
    """Map each offending file -> set of disallowed external hosts it loads."""
    permitted = allowed | SELF_HOSTS
    violations: dict[str, set[str]] = {}
    for path in site_dir.rglob("*"):
        if path.suffix == ".html":
            found = _hosts_in_html(path.read_text(encoding="utf-8", errors="ignore"))
        elif path.suffix == ".css":
            found = _hosts_in_css(path.read_text(encoding="utf-8", errors="ignore"))
        else:
            continue
        bad = {host for host in found if host not in permitted}
        if bad:
            violations[str(path.relative_to(site_dir))] = bad
    return violations


def main(argv: list[str]) -> int:
    site_dir = Path(argv[1]) if len(argv) > 1 else Path("site")
    if not site_dir.is_dir():
        print(f"error: '{site_dir}' is not a directory (run `mkdocs build` first)", file=sys.stderr)
        return 2

    violations = find_violations(site_dir, ALLOWED_HOSTS)
    if not violations:
        print(f"✓ external-host audit passed (allowed: {sorted(ALLOWED_HOSTS)})")
        return 0

    print("✗ external-host audit FAILED — disallowed resource hosts found:\n", file=sys.stderr)
    offenders: set[str] = set()
    for file, hosts in sorted(violations.items()):
        offenders |= hosts
        print(f"  {file}: {', '.join(sorted(hosts))}", file=sys.stderr)
    print(
        f"\n{len(offenders)} disallowed host(s): {', '.join(sorted(offenders))}\n"
        "If a host is intentional, add it to ALLOWED_HOSTS in "
        "scripts/check_external_hosts.py. Otherwise remove the reference.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
