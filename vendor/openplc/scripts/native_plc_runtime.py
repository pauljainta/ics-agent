from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def _load_sourceless(module_name: str):
    workspace_root = Path(__file__).resolve().parents[4]
    candidate_paths = [
        workspace_root
        / "sphere-usecases"
        / "water-treatment"
        / "implementations"
        / "openplc"
        / "scripts"
        / "__pycache__"
        / f"{module_name}.cpython-310.pyc",
        workspace_root
        / "sphere-usecases"
        / "water-treatment"
        / "implementations"
        / "openplc"
        / "scripts"
        / "__pycache__"
        / f"{module_name}.py",
        Path(__file__).resolve().parent / "__pycache__" / f"{module_name}.cpython-310.pyc",
        Path(__file__).resolve().parent / "__pycache__" / f"{module_name}.py",
    ]
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        if candidate_path.suffix == ".pyc":
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
