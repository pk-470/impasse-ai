"""Refuse to run the suite against stale compiled extensions.

`scripts/build_native.py` writes the mypyc extensions next to their sources, and
an extension module shadows a same-named `.py`. So after editing `position.py` or
`ai.py` without rebuilding, the tests import the *previous* build: they pass, but
they tested code that is no longer in the tree. That is silent and costs real
time, so fail loudly instead.

The comparison is content-based, against the digests the build records in
`src/.native_build.json`. An mtime comparison does not work here: mypyc's cache
is content-based, so a source whose mtime moved without its bytes changing is
correctly not rebuilt, and an mtime check would call a fresh build stale forever.
"""

import hashlib
import json
import sysconfig
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE = SRC / "impasse"
MODULES = ("impasse/position.py", "impasse/ai.py")
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
STAMP = SRC / ".native_build.json"


def pytest_configure(config: pytest.Config) -> None:
    compiled = list(PACKAGE.glob(f"*{EXT_SUFFIX}")) + list(
        SRC.glob(f"*__mypyc{EXT_SUFFIX}")
    )
    if not compiled:
        return  # pure Python: the sources are what runs
    try:
        recorded = json.loads(STAMP.read_text())
    except (OSError, ValueError):
        recorded = {}
    stale = [
        module
        for module in MODULES
        if (SRC / module).exists()
        and hashlib.sha256((SRC / module).read_bytes()).hexdigest()
        != recorded.get(module)
    ]
    if stale:
        raise pytest.UsageError(
            "compiled extensions do not match their sources: "
            + ", ".join(stale)
            + "\nAn extension shadows the source, so this run would test OLD code."
            "\nRebuild with `python scripts/build_native.py`, or remove the"
            " extensions with `python scripts/build_native.py --clean`."
        )
