#!/usr/bin/env python3
"""Author/mentor verification: run the pytest suite against the REFERENCE SOLUTIONS.

The tests import the package name `celldetect`, which normally resolves to
the student scaffolds (all NotImplementedError). This script proves that the
tests and the solutions are mutually correct:

  1. copy solutions/celldetect_solutions/*  ->  <tmpdir>/celldetect/  (renamed
     package), and copy tests/ -> <tmpdir>/tests/;
  2. run pytest with cwd = <tmpdir> and <tmpdir> first on sys.path, so that
     `import celldetect` finds the SOLUTIONS copy.

Running from the tmpdir matters: pytest puts the rootdir/cwd side of things
first on sys.path, so running from python/ would let the student package
shadow the solutions. We sidestep the whole question by leaving python/
entirely — nothing on sys.path points at the student package.

Usage:  cd python && python run_solution_check.py
Exit status 0 iff every test passes. Deterministic (the tests are seeded).
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent
SOLUTIONS = PYTHON_DIR / "solutions" / "celldetect_solutions"
TESTS = PYTHON_DIR / "tests"


def main() -> int:
    import pytest  # imported here so the error message below can mention pip

    with tempfile.TemporaryDirectory(prefix="celldetect_check_") as tmp:
        tmp_path = Path(tmp)

        # 1. Solutions, renamed to the package name the tests import.
        shutil.copytree(
            SOLUTIONS, tmp_path / "celldetect",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        # 2. A pristine copy of the tests.
        shutil.copytree(
            TESTS, tmp_path / "tests",
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
        # 3. A conftest at the tmp root makes <tmpdir> pytest's rootdir and
        #    puts it on sys.path ahead of everything else.
        (tmp_path / "conftest.py").write_text(
            "# rootdir anchor for run_solution_check\n"
        )

        # Make `import celldetect` resolve to the solutions copy — and make
        # sure nothing else can win the race:
        #   * drop python/ (this script's dir) from sys.path,
        #   * forget any already-imported celldetect modules,
        #   * put the tmpdir first, and export PYTHONPATH for any subprocess.
        sys.path[:] = [p for p in sys.path
                       if Path(p or ".").resolve() != PYTHON_DIR]
        for name in [m for m in sys.modules if m.split(".")[0] == "celldetect"]:
            del sys.modules[name]
        sys.path.insert(0, str(tmp_path))
        os.environ["PYTHONPATH"] = str(tmp_path)

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            # Sanity check before testing: the celldetect that imports must be
            # the tmp copy, or every result below would be meaningless.
            import celldetect
            origin = Path(celldetect.__file__).resolve()
            if tmp_path not in origin.parents:
                print(f"FAIL: 'celldetect' resolved to {origin}, not the "
                      f"solutions copy under {tmp_path}")
                return 2
            print(f"checking solutions via: {origin.parent}")

            code = pytest.main(["tests", "-q"])
        finally:
            os.chdir(cwd)

    print()
    if code == 0:
        print("PASS: all tests pass against the reference solutions.")
    else:
        print(f"FAIL: pytest exit code {code} — tests and solutions disagree.")
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
