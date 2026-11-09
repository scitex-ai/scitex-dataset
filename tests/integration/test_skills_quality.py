"""Enforces SciTeX skills quality checklist §1–§4.
Canonical: src/scitex/_skills/general/21_scitex-package-quality-checklist.md
"""

from pathlib import Path

from scitex_dev._skills_quality_pytest import make_skill_quality_tests

test_skills_quality = make_skill_quality_tests(
    package_root=Path(__file__).resolve().parents[2]
)


def test_skills_quality_wiring_returns_collectable_test():
    """The factory must hand back a callable pytest can collect.

    The real assertions live inside the generated test (see
    ``scitex_dev`` skills-quality helper); this guards the wiring
    itself so the module is not assertion-free (PS-206b).
    """
    # Arrange: factory output bound at module import time.
    # Act: nothing to invoke — collection imports this module.
    # Assert: pytest collects a callable under the `test_` name.
    assert callable(test_skills_quality)
