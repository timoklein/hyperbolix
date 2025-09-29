# Phase 6 – Cleanup & Documentation

## TODOs
- [ ] Remove Torch-specific code paths, parameters, and dependency entries once JAX parity is achieved.
- [ ] Delete compatibility flags and deprecated APIs; update `src/__init__.py` exports to point exclusively to JAX implementations.
- [ ] Refresh docs: rewrite `README.md`, `.codex/AGENTS.md`, and examples to reflect the new stack and usage patterns.
- [ ] Update CI/CD to install only JAX tooling and drop Torch jobs; ensure wheels/notebooks build cleanly.
- [ ] Prepare release notes summarising breaking changes, migration guidance, and known limitations for downstream teams.
- [ ] Coordinate final verification: run full `uv run pytest` suite, lint/format commands, and optional integration notebooks before tagging the release.

## Notes
- Communicate timelines and deprecation notices to collaborators well in advance; archive Torch-era tags or branches if long-term support is required.
- Consider packaging migration utilities (e.g. conversion scripts) separately if they remain useful post-cleanup.
