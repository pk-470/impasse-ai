"""
Compile the engine's hot modules to C extensions with mypyc, in place.

The extensions land next to their sources in `src/impasse/`. Python's import
system prefers an extension module over a same-named `.py` in the same
directory, so the editable install picks the compiled engine up with no
reinstall — and `--clean` removes them to go back to pure Python. Nothing else
in the project depends on whether this has been run.

Only `position.py` and `ai.py` are compiled, and both in a single mypyc
invocation: modules compiled together share one group and call each other
directly in C, whereas separate invocations would route those calls back
through the Python API and give most of the speedup back. `gui.py` and
`play.py` stay interpreted — they are not hot, and keeping them out means
pygame never has to type-check.

Usage:
    python scripts/build_native.py            # compile
    python scripts/build_native.py --clean    # remove the extensions
    python scripts/build_native.py --check    # report what is in place
                                              # (exit 1 if a .pyd is stale)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE = SRC / "impasse"
# Compiled together on purpose: same mypyc group means direct C calls between them.
MODULES = ("impasse/position.py", "impasse/ai.py")
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
# Records the source digests the current extensions were built from.
STAMP = SRC / ".native_build.json"

# mypyc writes intermediate .obj paths under src/build/temp.<platform>/Release/build/.
# On Windows that whole path must stay under MAX_PATH (260) unless long paths are
# enabled, or cl.exe fails with the unhelpful:
#   fatal error C1083: Cannot open compiler generated file: '': Invalid argument
MAX_PATH_BUDGET = 260
_PROBE = (
    "build/temp.win-amd64-cpython-313/Release/build/__native_0123456789abcdef0123.obj"
)


def compiled_artifacts() -> list[Path]:
    """Every extension module and build directory this script can produce."""
    found = [p for p in PACKAGE.glob(f"*{EXT_SUFFIX}")]
    # The shared mypyc group module lands at the top of `src`, not in the package.
    found += [p for p in SRC.glob(f"*__mypyc{EXT_SUFFIX}")]
    return found


def check_path_budget() -> None:
    """Warn if the build path is near the Windows MAX_PATH limit."""
    if sys.platform != "win32":
        return
    worst = len(str(SRC / _PROBE))
    if worst >= MAX_PATH_BUDGET:
        sys.exit(
            f"error: the build path would reach {worst} characters, over Windows'\n"
            f"       {MAX_PATH_BUDGET}-character limit, and cl.exe will fail with a\n"
            f"       misleading C1083 error. Move the project somewhere shorter, or\n"
            f"       enable Win32 long paths.\n"
            f"       path: {SRC / _PROBE}"
        )
    if worst > MAX_PATH_BUDGET - 40:
        print(
            f"warning: build paths reach {worst} of {MAX_PATH_BUDGET} characters; "
            f"deeper nesting will break the build.",
            file=sys.stderr,
        )


def clean() -> int:
    """Remove the compiled extensions and mypyc's build tree."""
    removed = 0
    for path in compiled_artifacts():
        path.unlink()
        print(f"removed {path.relative_to(SRC)}")
        removed += 1
    if STAMP.exists():
        STAMP.unlink()
        print(f"removed {STAMP.relative_to(SRC)}")
        removed += 1
    for tree in (SRC / "build", SRC / ".mypy_cache"):
        if tree.is_dir():
            shutil.rmtree(tree)
            print(f"removed {tree.relative_to(SRC)}/")
            removed += 1
    if not removed:
        print("nothing to clean; already pure Python")
    return 0


def source_digests() -> dict[str, str]:
    """sha256 of each compiled module's current source."""
    return {
        module: hashlib.sha256((SRC / module).read_bytes()).hexdigest()
        for module in MODULES
        if (SRC / module).exists()
    }


def stale_sources() -> list[str]:
    """
    Modules whose source differs from what was last compiled.

    Content-based, not mtime-based. mypyc's own cache is content-based, so a
    source whose mtime moved without its bytes changing is correctly *not*
    rebuilt -- an mtime comparison would then report a fresh build as stale
    forever. The build writes STAMP after a successful compile; this compares
    against it.

    An extension module shadows a same-named `.py`, so running against an
    out-of-date build silently executes old code: tests pass on the previous
    build and benchmarks measure a tree that is no longer there.
    """
    if not compiled_artifacts():
        return []
    try:
        recorded = json.loads(STAMP.read_text())
    except (OSError, ValueError):
        # Compiled artifacts with no usable stamp: cannot prove they are current.
        return list(MODULES)
    current = source_digests()
    return [m for m in MODULES if current.get(m) != recorded.get(m)]


def check() -> int:
    """Report whether the compiled engine is in place, and which flavour is live."""
    found = compiled_artifacts()
    if not found:
        print("pure Python: no compiled extensions in src/", flush=True)
    else:
        print("compiled extensions in place:", flush=True)
        for path in sorted(found):
            print(
                f"  {path.relative_to(SRC)}  ({path.stat().st_size:,} bytes)",
                flush=True,
            )
        stale = stale_sources()
        if stale:
            print(flush=True)
            for module in stale:
                print(
                    f"  STALE: {module} differs from what was last compiled",
                    flush=True,
                )
            print(
                "  The extension shadows the source, so this is running OLD code.",
                flush=True,
            )
            print("  Rebuild (python scripts/build_native.py) or --clean.", flush=True)
            return 1
    # Import in a subprocess so this script never holds an extension open (Windows
    # locks a loaded .pyd, which would make a later --clean fail).
    probe = (
        f"import sys; sys.path.insert(0, r'{SRC}');"
        "from impasse.position import NATIVE;"
        "print('pathfinders:', 'native (mypyc)' if NATIVE else 'pure Python')"
    )
    # A probe that cannot import the extension means the build is unusable.
    return 1 if subprocess.run([sys.executable, "-c", probe], check=False).returncode else 0


def build() -> int:
    """Type-check, then compile the hot modules in place."""
    check_path_budget()
    # mypyc type-checks as part of compiling, but a plain mypy run first gives
    # readable errors instead of a compile abort.
    print(f"type-checking {' '.join(MODULES)} ...", flush=True)
    mypy = subprocess.run(
        [sys.executable, "-m", "mypy", *MODULES], cwd=SRC, check=False
    )
    if mypy.returncode:
        return 1
    print("compiling with mypyc ...", flush=True)
    mypyc = subprocess.run(
        [sys.executable, "-m", "mypyc", *MODULES], cwd=SRC, check=False
    )
    if mypyc.returncode:
        return 1
    STAMP.write_text(json.dumps(source_digests(), indent=2))
    print()
    return check()


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--clean", action="store_true", help="remove the extensions")
    group.add_argument("--check", action="store_true", help="report what is in place")
    args = parser.parse_args()
    if args.clean:
        return clean()
    if args.check:
        return check()
    return build()


if __name__ == "__main__":
    sys.exit(main())
