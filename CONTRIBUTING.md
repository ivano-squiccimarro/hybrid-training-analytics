# Contributing

This is an experimental project under private review. Keep the repository private;
publication needs a new, explicit decision by the owner.

1. Install with `uv sync --locked --python 3.12`.
2. Make a small, scoped change. Preserve unrelated local work.
3. Add meaningful synthetic cases for parser/metric changes; never use a real FIT.
4. Run `uv run --locked python scripts/check.py`.
5. For report changes, generate a new demo and inspect units, missing values and
   limitations. For schema changes, document compatibility/migration explicitly.
6. Inspect every staged file. Exclude local data, reports, tokens and environments.

Do not interpret user-declared RPE/RIR as measurements or exercise suggestions
as confirmed labels. Do not prescribe training progression. No general refactors
or new dependencies merely to improve presentation.

The local interface currently uses Italian labels. UI translation, CI on other
platforms, a license choice and independent evaluation are future work, not
prerequisites for trying the synthetic examples.
