"""Smoke test for Phase 0 dependencies."""


def test_pydantic_imports():
    """Verify pydantic is importable and has a version."""
    import pydantic
    assert pydantic.__version__ is not None


def test_numpy_imports():
    """Verify numpy is importable and has a version."""
    import numpy
    assert numpy.__version__ is not None
