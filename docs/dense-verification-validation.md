# Dense temporal verification: validation experiment

Protocol recorded before the dense-stage results, 8 October 2026.

The previous continuous UT regression identified a downstream verification
bottleneck. This round changes sampling only: the Qwen model, semantic event
requirements, generation settings and conservative Gemma prompt threshold
remain unchanged. It uses recordings 13–16 (validation), not recordings 17–20
used in the previous test regression. No training or ground-truth crop is used.

For each of the 62 frozen Gemma candidates:

1. Verify up to eight uniformly sampled frames of the original six-second unit.
2. Keep a valid positive without retrying it.
3. For other units, verify eight frames of the first half, then the second half
   if still nonpositive. Halves cover the whole original interval, including
   short tails. A positive ends retries; failures remain pending unless another
   attempt finds positive evidence.

The experiment actually executes this conditional cascade; request cost is not
inferred from separately run pairs. Every request and raw broker reply is saved.
Labels are used only for scoring, never in request payloads or sampling times.
Coverage credits the interval of the view that produced a valid positive, not
the original wider candidate interval. Positive coverage is not proof of correct
action localization or calibrated confidence.

Selection rule: retain as an experimental option only if labeled event coverage
improves over uniform sampling without increasing the count of positive views
outside annotated target/ignored interactions. Report both costs even if this
rule fails. This small validation cannot certify precision, hourly false alarms,
or the full production acceptance gates. No automatic default or validation lock.

Run from the experimental worktree with its inference Python and backend on
PYTHONPATH:

```powershell
python -m evaluation.embeddinggemma2.validate_dense_verification
```

Ignored raw artifacts: `scratch/event-retrieval/dense-validation/results.json`.
The script refuses to overwrite an existing run. Shared candidate provenance and
official labels come from `scratch/event-retrieval/continuous-ut/results.json`.
Dataset source and its selected/other interaction distinction are documented in
[the continuous protocol](continuous-screening-experiment.md).

## Measured validation results

| Metric | Uniform | Adaptive dense cascade |
|---|---:|---:|
| Official selected events covered by a positive view | 3/8 (37.5%) | 6/8 (75%) |
| Events with positive-view temporal IoU >= 0.3 | 3/8 | 6/8 |
| Positive views outside target/ignored interactions | 0 | 0 |
| Positive views | 4 | 13 |
| Verification views submitted | 62 | 175 |
| Private Qwen HTTP requests, including caption requests | 66 | 188 |
| Sum of measured broker verification-batch time | 163.54 s | 464.89 s |

The complete adaptive run took 563.36 seconds including extraction and other
client overhead. Uniform batch time excludes extraction and client overhead;
it is not a separate end-to-end runtime measurement. Stage counts were 62 full
views, 58 first halves, and 55 second halves. No additional model training or
new Gemma embeddings were required: the frozen validation candidates were reused.

Final candidate states: 13 positive, 47 negative and 2 pending. Pending cases
retain invalid/failed evidence and are not counted as successfully ruled out.
The two missed selected targets were punches in seq13 (30.67–32.37 s) and seq14
(58.57–60.87 s). The improvement is three additional covered targets; positive
view count must not be confused with unique detected-event count.

Whole-card sampled GPU peak was 11.289 GiB, including other applications, and
Qwen process-tree sampled peak RSS was 7.621 GiB. These are 0.5-second sampled
peaks, not per-model CUDA allocations. Gemma was unloaded by the broker before
verification, as in the earlier experiment.

Decision: the preregistered experimental selection rule passes (higher coverage,
no increase in out-of-interaction positives). Retain the reproducible cascade
as an experimental verifier option. It costs approximately 2.85 times as many
private HTTP requests and still falls below the requested final recall gate of
85%. Do not activate it as the application default or mark validation passed.
Zero outside-interaction views in four recordings does not establish precision
or hourly false-alarm performance: overlapping views, partially overlapping
actions and officially ignored interactions require separate assessment.

45 targeted tests passed (dense sampling/state/interval checks plus continuous,
reference and worker regressions). Raw video, cached frames and measured replies
remain ignored. No test-set rerun or production-policy modification in this round.
The graph CLI is unavailable in this environment; the new evaluation modules
remain outside the existing main-checkout graph.
