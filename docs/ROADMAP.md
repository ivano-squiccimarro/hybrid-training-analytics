# Roadmap and current status

This is a usable research prototype, not a finished coaching product.

| Area | Implemented | Still open |
|---|---|---|
| Running FIT | CRC validation, UTC/null handling, offline reports | Wider device/export compatibility |
| Treadmill trace | Separate proportional-distance derivative, optional model GAP | Uniform-error assumption is not a validated per-sample correction |
| Signals and decoupling | Time support, pauses/gaps, explicit conditions and phase comparison | Independent physiological validation |
| Strength data | CSV diary, per-exercise volume and matching conditions | Assisted-load mode and broader exercise vocabulary |
| Partial strength sessions | FIT/manual session, optional annotations, zero known sets | Real-world usability study |
| Reconciliation | Motion hypotheses where available, device hints, optional split/merge | Real recognition accuracy; accelerometer may be absent |
| Suggestions | Explainable alternatives, prior context, explicit abstention | Calibrated probabilities and generalization |
| Evaluation | Frozen trials, separate labels, baselines, null denominators/cohorts | New remembered sessions and independent references |
| Hybrid context | Time relationships and separate proxies, identity-based deduplication | Causality, recovery and physiological fatigue are not claimed |

## Future validation and maintenance

- Review new source, screenshots and examples for privacy before every upload.
- Review scientific wording and limitations; do not turn test counts into accuracy claims.
- Licensing is defined: MIT for software and CC BY 4.0 for documentation;
  retain the scope in LICENSING.md and review third-party terms for future additions.
- Verify installation and UI on a second computer/platform.
- Check accessibility, touch/drag and language needs.
- Keep the project presented as an experimental portfolio prototype; wider
  usability and platform checks are future work, not claims of current readiness.

CI, localization, trained models and expanded interfaces are optional later work.
None is required by the current synthetic quick start.
