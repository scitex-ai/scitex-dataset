# Skill resource boundaries

`test_skill_resource_boundary.py` exercises the real Click CLI and FastMCP
`skills_get` handler against per-test copies of the bundled package. Owned
marker files outside the skill directory cover parent paths, absolute paths,
escaping symlinks and backslash names. Named pages, including `SKILL`, must
retain their exact content. No remote service, database or model is used.

The adapters share `src/scitex_dataset/_skill_resources.py`. Its exact-name
lookup does not interpret user input as a path, and resolved resources must
stay within the declared directory. This protects skill reads, not arbitrary
local filesystem access or concurrent modifications by an actor able to rewrite
the installed package. The separate skill-install command is outside this check.
