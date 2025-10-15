"""Benchmark suite for hyperbolix JAX implementation.

Run benchmarks:
    uv run pytest benchmarks/ --benchmark-only

Save baseline:
    uv run pytest benchmarks/ --benchmark-only --benchmark-save=baseline

Compare to baseline:
    uv run pytest benchmarks/ --benchmark-only --benchmark-compare=baseline

Generate reports:
    uv run pytest benchmarks/ --benchmark-only --benchmark-histogram
"""
