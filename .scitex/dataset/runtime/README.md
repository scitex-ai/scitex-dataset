# `.scitex/dataset/runtime/`

Regenerable per-host state for `scitex-dataset` — downloaded snapshots,
caches, log files. Everything here is reproducible from the package and
`config.yaml`, so it is git-ignored.

The dataset INDEX is deliberately not here. It lives in the shared SciTeX
store (`scitex_dev.store`), because an index kept as a per-host file is
state with no owner: whoever can open the file holds every permission, and
each host ends up with its own answer to the same question.

See the SciTeX `general/01_ecosystem_06_local-state-directories` skill
for the canonical layout (project-scope wins over user-scope; `SCITEX_DIR`
relocates user-scope atomically).
