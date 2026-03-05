"""
Smoke test for the Chronos inference pipeline.

A smoke test is the most basic level of testing — it simply verifies that the
system can execute at all without crashing. This file acts as a "does it even
start?" sanity check for the Chronos test suite.

In a full CI/CD pipeline this would be accompanied by:
  - Unit tests for feature extraction (extractor.py)
  - Integration tests for the FastAPI /predict, /forecast, /patterns, /recommend
    endpoints
  - Performance benchmarks with the synthetic data generators

For now, the single assertion below ensures pytest discovers at least one passing
test, which keeps the test-runner green while the project is bootstrapped.
"""


def test_smoke():
    """Minimal smoke test: verify that basic Python arithmetic works.

    This trivial assertion confirms that the test infrastructure (pytest, CI
    runner, virtual-env) is wired up correctly. If this test fails, something
    is fundamentally broken in the environment rather than in Chronos code.
    """
    assert 1 + 1 == 2  # Baseline sanity check — should always pass
