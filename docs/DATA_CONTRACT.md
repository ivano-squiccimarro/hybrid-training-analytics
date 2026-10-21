# Data contract overview

Schemas are version 1. Incompatible changes need a version and migration; new
artifacts are additive. Detailed field validation lives in the typed/domain
modules and synthetic tests, not inferred from a private example dataset.

## Running

RunData includes source SHA-256, schema version, explicit display timezone,
synthetic flag, warnings, timer pauses and samples. Samples contain:

| Field | Meaning |
|---|---|
| timestamp_utc | UTC ISO 8601 with offset |
| elapsed_s | Seconds since the first valid record |
| heart_rate_bpm | Recorded HR or null |
| speed_m_s | Device speed estimate or null |
| distance_m | Device cumulative distance estimate or null |

Naive external timestamps are rejected. Decoder FIT timestamps are treated as UTC,
never as machine-local time. Optional cadence/power and user annotations are
sidecars, not rewrites of RunData. Derivative scaling has its own artifact type.

## Strength CSV

UTF-8, comma-separated, dot decimal; use the exact header in
`examples/strength_template.csv`. One row per declared set:
`timestamp,session_id,exercise,load_mode,body_mass_kg,added_load_kg,external_load_kg,set_index,reps,rir,rpe`.
Timestamps have offsets; mode is external or bodyweight. Mass/load values use kg;
reps and set_index are positive integers; optional RIR/RPE stay blank when unknown.
Assisted negative loads are unsupported. Keep the load convention consistent
(e.g. combined mass of both dumbbells) and distinguish exercise variants/technique.

## Minimum strength session

`artifact_type=strength_session_summary`, independent of CSV or full reconciliation.
Identity is `fit-<SHA256>` or `manual-<UUID32>`; source_ref is the FIT hash or null.
UTC bounds may be unknown. Recorded bounds/elapsed duration, declared duration
and HR support seconds remain separate, with per-field provenance.

Declarations: optional body_scope (upper/lower/full_body), session_rpe 1..10,
sensor (wrist/chest/other), start/end together, notes, remembered exercise IDs,
selected confirmed set IDs and explicit CSV linkage. UX seconds/actions require
local opt-in; missing measurements stay null. No sensor/RPE inheritance from runs.

Selected sets carry source/revision/slot references and confirmed snapshots.
Remembered names contain no implied reps/kg. acknowledged means a saved minimum
session, not confirmation of unknown fields. Later edits append revisions with
an expected parent; they do not overwrite raw data or older revisions.

## Reconciliation and evaluation

Raw parsed FIT, quality summaries, algorithm hypotheses, user-confirmed slots
and frozen suggestion trials are separate artifacts. Raw FIT storage is private.
Snapshots record source/configuration/code hashes. A finalized detailed export
has stricter rules than a minimum-session JSON; no fake CSV set is created.

Trial labels store optional remembered exercise/family, usefulness and declared
corrections separately. Only latest labels enter aggregate metrics; unknown labels
are not errors. Copied JSON markers cannot authenticate real-world truth on their own.

## Storage and rendering

`data/processed/hta-local/` holds private catalogs and revision files; reports go
to a fresh chosen output directory. Neither belongs in Git. Strings in HTML are
escaped. The server has authenticated API routes and explicit output allowlists,
not a general filesystem server. Session tokens are local and not committed.
