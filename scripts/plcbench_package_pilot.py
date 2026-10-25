"""Package only the preserved 30 PLCBench pilot bundles for private handoff.

The archive is intentionally separate from Git. It is deterministic for the
same input bytes, source commit, and config hashes. Review its contents before
sharing; this script does not certify that arbitrary text is free of secrets.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "wd-batch": "pilot_wd_loss_v1.json",
    "wt-lot": "pilot_wt_loss_v1.json",
    "wt-dosing": "pilot_wt_dosing_v1.json",
}
BUNDLE_FILES = {"history.txt", "meta.json", "score.json", "trace.csv", "transactions.json"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _collect(run_root: Path) -> dict[str, bytes]:
    contents: dict[str, bytes] = {}
    for group in GROUPS:
        directory = run_root / group
        expected = {f"{arm}-{n}" for arm in ("matched", "native") for n in range(1, 6)}
        actual = {item.name for item in directory.iterdir() if item.is_dir()}
        if actual != expected:
            raise ValueError(f"{group}: expected 10 pilot bundles; found {sorted(actual)}")
        if {item.name for item in directory.iterdir() if item.is_file()} != {"comparison_table.txt"}:
            raise ValueError(f"{group}: unexpected or missing top-level files")
        paths = [directory / "comparison_table.txt"]
        for bundle_name in sorted(expected):
            bundle = directory / bundle_name
            if bundle.is_symlink():
                raise ValueError(f"{bundle}: bundle directory must not be a symlink")
            if {item.name for item in bundle.iterdir()} != BUNDLE_FILES:
                raise ValueError(f"{bundle}: unexpected or missing bundle files")
            paths.extend(bundle / name for name in sorted(BUNDLE_FILES))
        for path in paths:
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"{path}: expected an ordinary file")
            contents[(Path("runs") / path.relative_to(run_root)).as_posix()] = path.read_bytes()
    return contents


def _add_bytes(archive: tarfile.TarFile, path: str, data: bytes) -> None:
    info = tarfile.TarInfo(path)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def package(run_root: Path, output: Path, source_commit: str) -> str:
    files = _collect(run_root)
    config_files = {
        f"configs/plcbench/{config}": (ROOT / "configs/plcbench" / config).read_bytes()
        for config in GROUPS.values()
    }
    manifest = "".join(f"{_sha256(data)}  {name}\n" for name, data in
                       sorted({**files, **config_files}.items())).encode()
    readme = (
        "# PLCBench 30-pilot evidence\n\n"
        f"Source checkout commit: `{source_commit}`. This archive has 30 pilot bundles "
        "and three comparison tables, with original bytes unchanged. The three "
        "versioned pilot objective configs are in the code checkout and hashed in "
        "`PILOT_MANIFEST.sha256`. No deployment keys, qualification records, or "
        "measured 50-pair results are included.\n\n"
        "Extract into the repository root, then run:\n\n"
        "```sh\nshasum -a 256 -c PILOT_MANIFEST.sha256\n"
        "PYTHONPATH=. python -m ics_agent.plcbench_evidence --pilot-root runs\n```\n\n"
        "Expected result: 30 episodes, 30 matching original verdicts, "
        "0 unassessable using the external versioned pilot objectives. The pilot "
        "is exploratory; menu semantics, information, execution, and budget "
        "differed between arms. See `runs/COMPARISON_SUMMARY.md` in the code "
        "checkout.\n"
    ).encode()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    with output.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                _add_bytes(archive, "PILOT_README.md", readme)
                _add_bytes(archive, "PILOT_MANIFEST.sha256", manifest)
                for path, data in sorted(files.items()):
                    _add_bytes(archive, path, data)
    return _sha256(output.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", help="code commit used to interpret the pilot")
    args = parser.parse_args()
    source_commit = args.source_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(package(args.run_root, args.output, source_commit))


if __name__ == "__main__":
    main()
