"""Run one import statement in a clean interpreter with some packages blocked.

Usage: python _import_probe.py <comma-separated blocked packages> <statement>

The last line of stdout is a JSON report containing the names of every loaded module
and the ModuleNotFoundError raised by the statement, if any. Blocking a package
simulates an environment where it is not installed. Used by test_backend_imports_2.py.
"""

import importlib.abc
import json
import sys


class BlockPackages(importlib.abc.MetaPathFinder):
    def __init__(self, blocked: set[str]) -> None:
        self.blocked = blocked

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in self.blocked:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def main() -> None:
    blocked = {name for name in sys.argv[1].split(",") if name}
    statement = sys.argv[2]
    sys.meta_path.insert(0, BlockPackages(blocked))

    error = None
    try:
        exec(statement, {})
    except ModuleNotFoundError as e:
        error = {"name": e.name, "message": str(e)}

    print(json.dumps({"modules": sorted(sys.modules), "error": error}))


if __name__ == "__main__":
    main()
