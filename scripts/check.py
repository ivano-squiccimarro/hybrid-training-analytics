"""Run the documented local checks without shared temporary directories."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from uuid import uuid4


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    temp = root / "work" / f"pytest-{uuid4().hex}"
    temp.parent.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("UV_CACHE_DIR", str(root / "work" / "uv-cache"))
    commands = [
        ["uv", "lock", "--check"],
        ["uv", "run", "--locked", "ruff", "check", "."],
        ["uv", "run", "--locked", "ruff", "format", "--check", "."],
        [
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--basetemp",
            str(temp),
        ],
    ]
    for command in commands:
        print("Checking:", " ".join(command), flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
