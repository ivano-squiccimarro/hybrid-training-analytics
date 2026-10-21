# Optional local trial protocol

The tools are implemented; a remembered real session and usability evaluation
are still needed. Nothing here requires uploading an activity.

1. Record a new session with a few independently remembered references: exercise,
   set/order, actual reps, load convention in kg, and approximate start/end times.
   Keep these notes before looking at suggestions. An old forgotten workout cannot
   supply recognition ground truth. Unknown boundaries/labels should remain unknown.
2. Export the original FIT/ZIP locally. Read its quality report. Save a minimum
   session even with no known sets; do not infer missing declarations from HR.
3. Optional timing in the minimum form starts only after explicit opt-in. It counts
   elapsed seconds and clicks there, not all keyboard/editor activity or total usage.
   An under-30-second workflow is an unvalidated product objective, not a result.
4. Select an unassigned/unconfirmed block and freeze its suggestions before labeling.
   The first capture is reused; target-session history/adjacency is excluded.
5. Label only what is independently remembered. Exercise/families, usefulness and
   correction counts are optional. No memory means null, not an incorrect prediction.
6. Labels remain separate from the timeline and training history. Corrections append
   revisions; metrics use the latest label without increasing the sample size.
7. Export evaluation JSON or run strength-evaluation-report with a new output path.
   Store personal outputs only in ignored local directories.

| Descriptor | Denominator |
|---|---|
| Suggestion coverage / abstention | All captured blocks |
| Top-3 hits including abstention | Blocks with a known remembered exercise |
| Top-3 conditional hits | Known exercise blocks where suggestions were made |
| Family overlap | Blocks with at least one independently declared family |
| Garmin / context baseline hits | Same known exercise blocks as the full ranking |
| Corrections / usefulness | Available declarations, with observation counts |
| UX time / actions | Latest opted-in observations, separate denominators |

Zero denominators give null. Garmin baseline uses the first three mapped native
names, including hints suppressed by the quality-aware ranking; it is not a
replica of Garmin's classifier. The context baseline omits native hints and
current-session adjacency. Keep synthetic and real cohorts separate.

Labels are memories, not independently instrumented ground truth. Report case
selection and block/session counts; do not generalize from one person. Manual
history with misleading earlier dates can still leak target answers. There is
no calibrated probability, preset success threshold or causal claim.

An optional future check is comparing private original exports before/after a
remembered Connect edit. Hash changes alone do not prove that labels/reps/loads
were exported. This compatibility check is not yet verified; automatic Garmin
synchronization or write-back is not provided.
