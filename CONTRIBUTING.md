# Contributing

## Development Setup

Use Python 3.11 or newer and `uv`:

```bash
uv sync --extra dev
uv run pytest -q
```

The optional `analysis` extra provides SciPy, Matplotlib, and Pillow. SciPy is required when LPP connected-component filtering is enabled with `min_patch_size > 1`.

## Changes to Numerical Code

- Add or update tests for equations, boundary handling, numerical tolerances, output policy, or LPP metric behavior affected by the change.
- Keep backend selection explicit. Do not report CPU execution as GPU execution.
- For scientific or performance claims, include the command, configuration, device/runtime metadata, and a reproducible report. Separate setup/compile time from steady-state timing.
- Do not add plots or paper-figure code to compute modules.
- Keep changes to the original paper-code repository out of this project.

## Before Opening a Pull Request

```bash
uv run pytest -q
uv run python scripts/validate_specs.py
uv run python -m compileall -q src scripts benchmarks
uv build
```

Pull requests should state the user-visible behavior change, tests run, numerical impact, and any backend/device used. Attach benchmark evidence only when it can be reproduced and its provenance is known.

## Release Checklist

1. Confirm CI passes on all supported Python versions.
2. Update version and release notes.
3. Build and inspect the wheel and source distribution.
4. Install the wheel in a clean environment and run the documented smoke examples.
5. Tag the release and publish a GitHub release with tested backend scope and known limitations.
6. For a PyPI release, test the artifacts on TestPyPI before production upload.
