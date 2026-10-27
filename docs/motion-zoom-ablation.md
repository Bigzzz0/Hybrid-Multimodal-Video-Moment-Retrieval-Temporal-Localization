# Motion-focused temporal retry

Protocol recorded before model inference, 8 October 2026. Spatial-only person
cropping did not improve development coverage. Test temporal selection separately.

Use the same full-frame temporal-v1 development control, seq1/2 (41 windows).
Keep its valid positives. For other units with at least two source frames, decode
approximately 8 FPS on CPU, reduce to grayscale 160x108, and compute the mean of
the strongest 5% of absolute inter-frame pixel differences after subtracting the
median global change. Select the highest-energy two-second span, with a fixed
central tie-break. Quiet traces with peak energy below 1 gray level do not trigger
a retry. Motion is only a sampling cue; it is not an anomaly decision.

Resample eight full-resolution frames in that selected two-second span and use
the same Qwen temporal-v1 prompt/model/budget. No crop, labels or annotation times
guide selection. Record motion traces, chosen intervals and actual request frames.
Only positive views count toward coverage, and only their actual view intervals
receive credit. Invalid original evidence remains pending unless a retry is
positive; a negative retry cannot erase uncertainty.

Promotion rule fixed before inference: strictly improve target coverage without
increasing positive views outside target/ignored interactions. Then freeze and
evaluate on validation. No acceptance lock or default activation from this small
experiment. Cached full-frame control means combined request cost is derived;
only retries are timed and executed in this run. Report this distinction.

This is a simple motion heuristic motivated by temporal sampling, not the
prompt-guided method or a reproduction of [Free Video-LLM](https://arxiv.org/abs/2410.10441).
Camera motion, walking and background motion can dominate the cue. Limitations
must remain visible rather than presenting motion as abnormality evidence.

Run `python -m evaluation.embeddinggemma2.compare_motion_zoom`. Immutable ignored
output: `scratch/event-retrieval/motion-zoom-dev/results.json`.

Development result before validation: full-frame temporal-v1 covered 2/4 targets;
motion-focused retry covers 3/4, with zero out-of-target positive views in both.
The previously missed push is recovered; the seq2 punch remains missed.
The fixed development promotion gate passes. Freeze the exact motion algorithm,
development artifact hash, candidate IDs and Gemma threshold before validation.

Validation uses the same 62 frozen Gemma candidate windows of seq13–16 as the
earlier dense validation. Actually run a matched temporal-v1 full-frame control,
then conditional motion retries; do not reuse the older legacy-prompt control
as if its prompt were identical. No cropping or parameter adjustment on validation.
The control and retries are both executed, but the generic retry report still
labels its combined totals derived; individual and summed times must be stated
separately. Do not compare eight validation targets directly with four development
targets. This is repeated validation regression, not a fresh blind test.

Run `python -m evaluation.embeddinggemma2.validate_motion_zoom`. Locked protocol,
control and raw replies are ignored under `scratch/event-retrieval/motion-zoom-validation`.
Test recordings 17–20 are not re-evaluated and no application default is activated.

## Measured results and decision

| Split | Full-frame temporal-v1 control | Control plus motion retry | Out-of-target positives |
|---|---:|---:|---:|
| Development, seq1/2 | 2/4 | 3/4 | 0 in both |
| Validation, seq13–16 | 3/8 | 5/8 | 0 in both |

Temporal IoU >=0.3 matches the same event counts in each arm. Validation retains
two pending units. Its remaining missed selected targets are punches in seq13
(30.67–32.37 s), seq14 (58.57–60.87 s) and seq16 (27–28.5 s). This strengthens
evidence that denser temporal detail helps some interactions, but does not show
that timing alone explains all failures. Motion can prioritize the wrong action;
Qwen can also reject visible motion as insufficient evidence of force/contact.

Development executed 37 retry views, 39 extra private HTTP requests and 106.34 s
of retry verification. Retry-phase wall time including preprocessing was 133.14 s.
With its cached full control, derived total requests are 82. These are not a newly
measured combined adaptive wall time.

Validation actually executed the full control (62 views, 65 private requests,
174.52 s) followed by retries (59 views, 66 requests, 184.41 s verification plus
preprocessing: 245.72 s retry-phase wall time). Sum of separately timed phase wall
times is 420.24 s; a single combined end-to-end timer was not used. Combined
request count is 131. Retry Qwen sampled whole-card GPU peak was 11.073 GiB,
including other applications; process-tree peak RSS was 9.168 GiB, sampled every
0.5 s. These are not per-model CUDA allocated/reserved bytes.

The within-arm improvement rule passes in validation, but **the prior dense-half
validation method remains higher in observed coverage: 6/8 versus 5/8**. The prior
method used a legacy prompt and 188 requests, so this is a historical reference,
not an isolated motion-versus-halves comparison. Do not replace it with this new
method as the most accurate option. Neither meets the final >=85% recall gate.
No default change, passed production lock, test rerun or merge.

62 targeted tests pass. Repaired evaluator interval-union duration to sort adaptive
views before merging, handle empty view sets and use the maximum view end rather
than submission order. This changes descriptive union/extent times, not coverage,
IoU, out-of-target counts, thresholds or promotion decisions. Raw artifacts are
retained; recomputed metrics are saved separately by
`python -m evaluation.embeddinggemma2.analyse_spatial_temporal` in ignored
`scratch/event-retrieval/spatial-temporal-analysis.json`. In particular validation
positive-view union is 22.80 s after repair, not the raw run's 20.20 s. Input-view
extent must not be used as certified normal-video hours for false-alarm rates.

## Next justified model comparison

Test a frozen action detector alongside Gemma and Qwen rather than continuing to
assume that spatial crops or more image prompts will solve the remaining misses.
[PyTorchVideo's official detection tutorial](https://pytorchvideo.org/docs/tutorial_torchhub_detection_inference)
provides pretrained AVA v2.2 video action models including SlowFast R50 detection,
with 32-frame input for SlowFast and person bounding boxes. This is a distinct
action model; Qwen can remain responsible for explanation and ambiguous review.

The [official AVA label map](https://dl.fbaipublicfiles.com/pytorchvideo/data/class_names/ava_action_list.pbtxt)
was fetched successfully and checked: fall down=5, fight/hit a person=64,
hand shake=68, hug a person=70, push another person=76. Object hit/push are separate
classes (34/46) and must not be confused with person interactions. AVA IDs are
one-based; the official tutorial prepends a background slot to predictions.
Fight/hit is broader than a specific punch: it remains evidence, not a guaranteed
punch confirmation. Multi-label classifier scores are not calibrated probabilities
for our deployment.

Start with an isolated, pinned prototype and existing cached person proposals;
the tutorial's Detectron2 example is not a requirement to replace all current
dependencies. Check compatibility and input transforms first. No action-detector
weights or inference were installed/run in this round. If later using GPU, add it
to the central broker and unload between Gemma, action detector and Qwen. Test
matched development normal/target clips, freeze thresholds before validation,
and measure calls, memory and latency before claiming improvement. Pretrained
availability means no weight retraining is necessary to begin this comparison;
cross-domain accuracy is still unproven.

Graph refresh remains unavailable because the graft executable is absent. Main
checkout graph context was read before scoping; new evaluation modules are unindexed.
