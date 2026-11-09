# Release checks

The active release workflow is
`.github/workflows/pypi-publish-and-github-release-on-tag.yml`.
It selects an existing canonical version tag explicitly through `refs/tags/`.
`.github/ci/check_release_version.py` checks that the selected tag, checked-out
commit and `pyproject.toml` version agree before test/build dependencies install.
The historical `exec-in-sif.sh` helpers are not called by this workflow.

The three Python test jobs each use a disposable PostgreSQL 16 service. Docker
assigns a free host port; the workflow validates that mapping and exports its
fixture DSN as `SCITEX_STORE_DSN`. It does not select an operator database or
assume that host port 5432 is free. These jobs run directly on their runner,
so the DSN uses `127.0.0.1`.

Tests use one normal `.[all,dev]` installation in `.venv`, followed by dependency
validation and `.venv/bin/python -m pytest tests/ -x`. Failed installation,
incompatible dependencies and a missing test directory stop the job. The build
job uses its own `.venv` for build tools and distribution creation. Neither
job installs packages into the runner's interpreter.

Behavioral controls are in
`tests/integration/test_release_workflow_integrity.py` (actual Bash control flow
with inert command fixtures), `test_release_tag_admission.py` (input/tag-ref
admission), and `test_release_source_version.py` (real owned Git tags/commits).
The inert fixtures do not establish database service health, dependency
resolution, scientific test results or publication. Require actual full-profile
CI and normal wheel/sdist/installed qualification before a new immutable tag;
published tags and versions must not be replaced. Promote accepted source through
the normal branch checks and review process before release.
## Shipped documentation qualification

The Sphinx configuration reads the selected checkout's `pyproject.toml`, even
when the package is not installed or the build starts from another directory.
The docs extra selects Sphinx 7/8 for sphinx-click's supported autodoc API;
docutils errors must remain visible. Keep source links at defining modules,
where re-export aliases cannot give an entire module the wrong backlink prefix.

After version or interface changes, build in the complete `all,dev,docs` profile:

```bash
sphinx-build -E -W -d docs/sphinx/_build/doctrees -b html docs/sphinx docs/sphinx/_build/html
```

Refresh `src/scitex_dataset/_sphinx_html/` from that output, preserving its
public asset ignore exceptions and keeping doctrees outside the bundle. Stage
the public HTML, source-link copies, fonts and icons. The checks in
`tests/integration/test_shipped_documentation.py` cover version/import identity,
tracked build inputs and representative rendered CLI commands. They do not
accept every prose claim, external link, public wheel or served web page.
