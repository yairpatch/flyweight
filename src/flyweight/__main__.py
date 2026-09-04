"""`python -m flyweight`, the invocation that does not depend on PATH.

The `flyweight` console script lands in the interpreter's Scripts directory,
which is not on PATH in two common Windows setups: a Microsoft Store Python,
whose per-user `LocalCache\\...\\Scripts` is never added, and any non-venv
`pip install --user`. pip prints a warning about it and installs anyway, so the
install genuinely succeeded and the command still could not be found.

Running the package as a module needs nothing but the interpreter that
installed it, so it is the answer to "it installed, now what" and the form the
README uses for the first command after an install.
"""
from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
