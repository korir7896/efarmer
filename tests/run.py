"""Dependency-free test runner: python -m tests.run"""

from __future__ import annotations

import traceback

from . import test_private, test_scaffold


def main() -> int:
    failures = 0
    for module in (test_scaffold, test_private):
        print(f"{module.__name__}:")
        for name in sorted(n for n in dir(module) if n.startswith("test_")):
            try:
                getattr(module, name)()
                print(f"  ok    {name}")
            except Exception:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
