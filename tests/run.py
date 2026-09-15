"""Dependency-free test runner: python -m tests.run"""

from __future__ import annotations

import traceback

from . import test_scaffold


def main() -> int:
    failures = 0
    for name in sorted(n for n in dir(test_scaffold) if n.startswith("test_")):
        try:
            getattr(test_scaffold, name)()
            print(f"  ok    {name}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
