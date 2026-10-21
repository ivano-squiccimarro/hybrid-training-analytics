# Methodology and limits

## Evidence and time

Heart rate is a recorded FIT channel, not guaranteed accurate. Indoor speed and
distance are device estimates; RPE, RIR, treadmill settings and final distance
are declarations. Derived metrics remain separate. JSON/CSV timestamps require
an explicit offset and normalize to UTC; display uses an explicit IANA timezone.
Null, zero, pauses and unknown timer state have distinct meanings.

## Temporal support

Metrics integrate trapezoids between valid adjacent records, clipping to a window
only inside such pairs. Pairs crossing missing data, gaps beyond the configured
limit or timer pauses are excluded, rather than filled. Each result exposes
valid seconds/coverage and reasons for unavailability. Coverage is not accuracy.
Technical quality checks (including abrupt HR changes or long unchanged HR) are
conservative heuristics, not diagnoses; thresholds are saved in configuration.

## Running outputs

The original report's HR change is `100 * (HR_second / HR_first - 1)` over equal
elapsed halves of an admissible constant phase. It is distinct from advanced
decoupling: `EF = mean(output) / mean(HR)` on common temporal support and
`D = 100 * (EF_first - EF_second) / EF_first`. Output is speed in m/s or power
in W, never pace with an ambiguous direction. Defaults include a 300 s warm-up,
600 s minimum phase and 10 s maximum gap; coverage, output/cadence consistency,
confirmed output and constant incline are checked. These are project parameters,
not validated individual HR zones or physiological decision thresholds.

## Treadmill derivatives

A declared target distance gives `k = target_distance_m / observed_endpoint_delta_m`.
The derivative scales relative cumulative distance and eligible device speed;
raw samples and timestamps remain unchanged. Uniform error is an assumption,
not proof of correct second-by-second speed. Pace from valid positive speed is
`1000 / speed_m_s` in s/km; pauses/unknowns stay unavailable.

Optional model GAP uses declared constant grade `g = incline_pct / 100` and
`pace * C(0) / C(g)`, with the Minetti running-cost polynomial in calibration.py.
No incline declaration means GAP is null, not assumed flat. It is a model estimate,
not a personally calibrated equivalent pace. The model reference is Minetti et al.,
Energy cost of walking and running at extreme uphill and downhill slopes (2002),
DOI: 10.1152/japplphysiol.01177.2001. Applicability is limited by the underlying
model and the uniform scaling/constant-grade assumptions.

## Strength and concurrent context

Known external volume is sum of `(external_load_kg + added_load_kg) * reps`,
kept per exercise. Bodyweight system-mass proxy includes body mass separately;
it is not mechanical work. No 1RM or actual neuromuscular fatigue is inferred.
Session RPE times explicit session minutes gives an AU proxy; set RPE is not
substituted for session RPE. Clock duration includes rests/forgotten periods.

A partial session can contain no known sets; complete-session volume remains
unknown. A remembered name alone creates no reps or load. FIT/CSV identities or
explicit links avoid double counting; temporal proximity alone never merges.
Sessions with unknown time are retained without an invented week. Temporal
association with a later run does not establish interference, causality or recovery.

## Recognition and evaluation

Motion clustering, when acceleration is exported, proposes work/rest boundaries.
Device markers are separate unverified hints. HR alone does not identify an
exercise. Rankings combine declared plans and eligible prior history with usable
native aliases; no evidence can result in abstention. Family names are product
vocabulary, not measured muscle activation. Lifting kg are not predicted from
wrist signals. Exact exercises and loads may remain unknown.

Frozen trial predictions precede labels. Evaluation excludes target-session
history/adjacency, uses remembered labels separately, and exposes denominators.
Synthetic and real cohorts stay separate. Labels are user declarations, not
independent ground truth; selection/recall/context leakage can still bias results.
See [the protocol](LOCAL_TRIAL_PROTOCOL.md). No recognition accuracy is established
by the passing tests.

## Reproducibility and implementation references

Inputs, parameters, units, schema and code/software hashes are recorded. New
revisions use an explicit increasing sequence without inventing UTC timestamps;
legacy tied timestamps retain their earlier fallback. Only one writer per workspace
is supported; the server lock does not coordinate separate processes.

Implementation dependencies: [fitdecode](https://github.com/polyvertex/fitdecode),
[Garmin FIT protocol](https://developer.garmin.com/fit/protocol/),
[Matplotlib](https://matplotlib.org/), [uv](https://docs.astral.sh/uv/).
These tools do not independently validate the project's methodological choices.
