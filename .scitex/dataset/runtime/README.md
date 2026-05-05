# `.scitex/dataset/runtime/`

Regenerable per-host state for `scitex-dataset` — local SQLite + FTS5
index, downloaded snapshots, log files. Everything here is reproducible
from the package and `config.yaml`, so it is git-ignored.

See the SciTeX `general/01_ecosystem_06_local-state-directories` skill
for the canonical layout (project-scope wins over user-scope; `SCITEX_DIR`
relocates user-scope atomically).
