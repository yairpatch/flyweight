<!--
Thanks for the contribution. Fill in what applies and delete what does not.
If your CI checks sit unstarted, that is GitHub holding workflow runs from
first-time contributors until a maintainer releases them -- not a failure.
-->

## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Why

<!-- The reasoning, or what broke. For a limitation being lifted, say which
     line of README.md#current-limitations it retires. -->

## How it was tested

- [ ] `ruff check src tests setup.py`
- [ ] `mypy src/flyweight`
- [ ] `pytest -q`
- [ ] `ctest --test-dir build/native --output-on-failure -C Release`
- [ ] Ran against a real model (say which below)

**Hardware and model:**
<!-- e.g. RTX 4070 12 GB, driver 560, Qwen3.5-27B UD-IQ2_XXS, 32K context,
     --backend cuda. Write "CPU backend only" or "not run" if that is the
     case -- an honest gap is more useful than a blank. -->

## Performance

<!-- Only if this is a perf change. Before/after numbers with the model,
     quantization, GPU and context length they were measured at. Numbers from
     one card do not transfer to another. -->

## Risk

<!-- Anything a reviewer should look at twice: a format the dispatch now
     handles differently, a kernel that assumes an alignment, a default that
     changed. If the change might help your hardware and hurt someone else's,
     it should sit behind a FLYWEIGHT_* variable defaulting to the old
     behaviour -- say so here. -->
