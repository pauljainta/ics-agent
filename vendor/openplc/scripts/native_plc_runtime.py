from __future__ import annotations

import importlib.machinery
import importlib.util
import marshal
import sys
from pathlib import Path


def _is_sourceless_payload(candidate_path: Path) -> bool:
    return candidate_path.suffix == ".pyc" or candidate_path.name.endswith(
        ".pyc_payload"
    )


def _is_compiled_wrapper(candidate_path: Path) -> bool:
    """Return True when a pyc is this loader wrapper compiled into itself."""
    if not _is_sourceless_payload(candidate_path):
        return False
    try:
        code = marshal.loads(candidate_path.read_bytes()[16:])
    except Exception:
        return False
    return "_load_sourceless" in getattr(code, "co_names", ())


def _load_sourceless(module_name: str):
    script_dir = Path(__file__).resolve().parent
    candidate_paths = [
        script_dir / "payloads" / f"{module_name}.cpython-310.pyc_payload",
        script_dir / f"{module_name}.cpython-310.pyc",
        script_dir / f"{module_name}.py",
        script_dir / "__pycache__" / f"{module_name}.cpython-310.pyc",
        script_dir / "__pycache__" / f"{module_name}.py",
    ]
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        if _is_compiled_wrapper(candidate_path):
            continue
        if _is_sourceless_payload(candidate_path):
            loader = importlib.machinery.SourcelessFileLoader(
                module_name, str(candidate_path)
            )
        else:
            loader = importlib.machinery.SourceFileLoader(
                module_name, str(candidate_path)
            )
        spec = importlib.util.spec_from_loader(module_name, loader)
        if spec is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        loader.exec_module(module)
        return module
    raise ModuleNotFoundError(f"unable to locate OpenPLC support module `{module_name}`")


def main() -> int:
    _load_sourceless("modbus_raw")
    runtime = _load_sourceless("native_plc_runtime")
    return int(runtime.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
