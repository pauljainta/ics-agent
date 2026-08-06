from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def _load_sourceless(module_name: str):
    pyc_path = Path(__file__).resolve().parent / "__pycache__" / f"{module_name}.cpython-310.pyc"
    loader = importlib.machinery.SourcelessFileLoader(module_name, str(pyc_path))
    spec = importlib.util.spec_from_loader(module_name, loader)
    if spec is None:
        raise ModuleNotFoundError(
            f"unable to create import spec for compiled module {module_name}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    loader.exec_module(module)
    return module


def main() -> int:
    bridge = _load_sourceless("modbus_bridge")
    return int(bridge.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
