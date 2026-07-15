# Contributing to pydreg

## Setup

```bash
git clone https://github.com/adamyhe/pydreg.git
cd pydreg
uv sync --extra gpu --group dev
uv run pytest tests/ -q
```

`uv sync --extra gpu` is safe on any platform: the `gpu` extra (CuPy) only
resolves on Linux with a matching architecture, and is a no-op elsewhere
rather than a failure.

## Tests

```bash
uv run pytest tests/ -q
```

The suite (40 tests as of this writing) covers each module in isolation
plus a full synthetic end-to-end pipeline run. Model-dependent tests are
skipped (not failed) if the Hugging Face repo hosting the pretrained
weights is unreachable.

## Before making changes

- **Algorithmic/structural changes**: read `docs/PLANNING.md` first. It's
  the comprehensive design record — module boundaries, exact algorithm
  specs, and, importantly, a full list of upstream R quirks that are
  reproduced deliberately rather than "fixed," because the pretrained
  model's expected output depends on them. Breaking one of those
  intentionally-kept quirks is a correctness regression, not a cleanup.
- **Performance changes**: read `docs/PERF_LOG.md` first, and append a new
  dated entry for any change you make. Its ground rule applies to any PR
  touching performance: a change must not alter the pipeline's output
  (same scores, same peaks), verified against the test suite and, ideally,
  a full CLI diff — "looks right" isn't sufficient. `docs/OPTIMIZATION.md`
  has the plain-language summary of the design choices already in place if
  you want the short version first.
- For a plain-language overview of the algorithm itself (not the
  implementation), see `docs/METHODS.md`.

## Pull requests

Keep changes scoped and include tests for new behavior. If a change
affects scores or peak calls in any way, say so explicitly and explain why
(e.g. a genuine bug fix vs. an intentional behavior change) — silent output
changes are the one thing this project can't tolerate, given how much of
it exists specifically to match a pretrained model's expected input/output
distribution.
