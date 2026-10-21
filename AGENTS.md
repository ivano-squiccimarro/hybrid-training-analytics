# Project working agreement

Read README.md, docs/ROADMAP.md and docs/METHODOLOGY.md before changing behavior.
Preserve existing work and inspect LAVORO_IN_CORSO.txt in the workspace and its
ancestors. Register scope before edits; never assume another writer's scope has
expired. One writer per directory; preserve unrelated processes and outputs.

## Scientific invariants

Keep timestamps in UTC with explicit offsets, and units explicit (s, bpm, m/s, m,
kg). Distinguish recorded channels, device estimates, user declarations and
inferences. Preserve None/null; no arbitrary filling or fabricated samples.
Never infer individual HR zones, VO2max, lifting kg from HR, training prescriptions
or medical conclusions. Coverage does not establish accuracy; volume is a proxy.

## Privacy and scope

Only synthetic fixtures/examples belong in Git. Never commit real FIT, personal
records, private notes, reports, tokens, caches or environments, even privately.
Do not upload, publish, change visibility, edit accounts or modify Git history
without explicit authorization for that action. This repository stays PRIVATE
until its owner explicitly approves publication. Use explicit staging paths.

Domain logic belongs under src/hybrid_training_analytics; keep I/O separate.
No new dependencies, framework changes or unrelated refactoring without a task
that requires them. HTML must escape user strings and work offline.

## Verification

uv sync --locked --python 3.12
uv run --locked python scripts/check.py
uv run --locked python -m hybrid_training_analytics demo --out reports/demo-new

Use fresh output and temporary directories; never delete or reuse shared pytest
basetemp. Report failures and skipped checks honestly. Update documentation and
your own work log entry to CONCLUSO when finished. A synthetic pass does not
validate recognition accuracy or usability with real people.
