# flyweight

- Every PR that changes what ships (`src/`, `native/` outside tests/bench/tools,
  `web/src/`, `pyproject.toml`, `setup.py`, `MANIFEST.in`) bumps `version` in
  `pyproject.toml`: patch for a fix, minor for a feature. Docs, plans, bench,
  tools and CI do not. `python tools/check_version_bump.py origin/main` is the
  check CI runs; run it before opening the PR.
- Releases are tag-driven: pushing `v<version>` builds the wheels, creates the
  GitHub release and publishes to PyPI. Nothing publishes without a tag.
- The native library in `src/flyweight/_native` is installed by
  `python -m flyweight.native_build`, never by copying over it.
