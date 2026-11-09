"""Verify public documentation bytes in built artifacts before publication."""

import json
import pathlib
import tarfile
import zipfile

MARKERS = {".gitignore", ".buildinfo", ".nojekyll"}


def check(source_root, distribution_root):
    root = pathlib.Path(source_root)
    bundle = root / "src/scitex_dataset/_sphinx_html"
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("Source documentation directory is missing or symbolic")
    expected = {}
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symbolic documentation input is not admitted")
        if path.is_file() and path.name not in MARKERS:
            expected[path.relative_to(bundle).as_posix()] = path.read_bytes()
    if not expected:
        raise ValueError("No public source documentation was checked")
    dist = pathlib.Path(distribution_root)
    wheels, archives = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise ValueError("Expected exactly one wheel and one source archive")
    with zipfile.ZipFile(wheels[0]) as wheel:
        if len(wheel.namelist()) != len(set(wheel.namelist())):
            raise ValueError("Duplicate wheel members are not admitted")
        for name, content in expected.items():
            member = "scitex_dataset/_sphinx_html/" + name
            if member not in wheel.namelist() or wheel.read(member) != content:
                raise ValueError("Wheel documentation missing or changed: " + name)
    with tarfile.open(archives[0]) as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate source archive members are not admitted")
        by_path = {name.split("/", 1)[1]: name for name in names if "/" in name}
        for name, content in expected.items():
            member = "src/scitex_dataset/_sphinx_html/" + name
            if (
                member not in by_path
                or archive.extractfile(by_path[member]).read() != content
            ):
                raise ValueError(
                    "Source archive documentation missing or changed: " + name
                )
    return {
        "public_docs_files": len(expected),
        "wheel": wheels[0].name,
        "sdist": archives[0].name,
    }


if __name__ == "__main__":
    source = pathlib.Path(__file__).resolve().parents[2]
    print(json.dumps(check(source, source / "dist"), sort_keys=True))
