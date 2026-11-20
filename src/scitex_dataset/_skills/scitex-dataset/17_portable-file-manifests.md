---
description: "[TOPIC] Portable file manifests. [DETAILS] Export known-file provenance with relative paths and replay checksum/size pins under another root, online or offline."
tags: [scitex-dataset-portable-file-manifests]
---
# Portable file manifests

Use `create_file_manifest(records, root)` to export FetchRecord objects or JSON
records. The `scitex-file-manifest/v1` schema retains URL, expected SHA256, size,
license/citation and extra fields such as checksum_basis. Destination paths are
relative POSIX paths; original fetch records are not modified. URL accessibility
and filesystem-specific filename constraints remain the caller's responsibility.

```python
import json
from scitex_dataset import create_file_manifest, replay_file_manifest

manifest = create_file_manifest(records, "/data/original")
# Persist with json.dumps(manifest); transport it independently of raw data.
verified = replay_file_manifest(manifest, "/scratch/new-root")
verified_offline = replay_file_manifest(manifest, "/scratch/new-root", offline=True)
```

All entries/paths are validated before writes. Absolute/parent traversal,
Windows drive paths, duplicate destinations and symlink escapes are rejected.
Existing mismatched files are preserved unless replace=True. Downloads are
staged and checked against both the declared hash and size before installation.
Offline mode verifies only existing files and cannot replace or download.
Installation is atomic per file; previously successful files remain after a
later network error and can be verified/reused on retry.

Pinned observed hashes establish byte identity, not publisher authenticity.
No checksums are automatically learned or updated on replay. A supplied
checksum_basis is preserved. The helpers perform no database/index writes.
