# Continuous labeled screening, 8 October 2026

Branch: `codex/embeddinggemma2-event-retrieval`. No model weights trained and no
application threshold automatically replaced.

## Why this experiment

The previous trimmed-clip reference experiment scored well within UT but
selected every window of Normal634. Its normal references were mostly short
greeting/pointing clips. This experiment uses annotated continuous recordings
to supply idle/walking/bystander context as well as greetings, and tests whether
removing the recording's average feature improves transfer between scenes.

Source: [official UT-Interaction](https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html).
The original recordings have 30 FPS, 720x480 pixels, with start/end frame labels
and actor boxes. Boxes are **not** used to crop inputs or give the model an oracle
location. Frame bounds become `[start/30, (end+1)/30)`; the original numbering
origin has a possible one-frame ambiguity, immaterial to broad six-second
coverage but relevant to exact boundaries. A development contact sheet at
25/26/27 seconds in seq1 visually confirms alignment with the annotated punch.

The existing segmented clips and browser conversions remain untouched. Original
continuous archives are stored in ignored `datasets/ut-full`:

- set1 SHA256: `172965ce7208704165448c6d8890d9853e9fee96ffc7e06036bb0ea1b6c29b2f`
- set2 SHA256: `8c79287321b2aaa0d0a348e9994fd3a9442fafce1f1ac1c8b28a7deef545e07e`

## Frozen protocol

- Development source recordings: seq1–12. Validation: seq13–16. Test: seq17–20.
- These sources follow the earlier split. Trimmed clips from test sources were
  evaluated previously: this is source-disjoint regression against the current
  reference bank, **not a new never-seen external test**.
- Officially selected push/punch instances are target events. Other annotated
  push/punch instances and kicks are excluded from normal-reference preparation
  and from non-target-window counting. The official competition's spatial
  detection metric is not reproduced here.
- Gemma BF16, pinned revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, 768-D.
  Six-second windows with three-second stride, including the short tail; video
  2 FPS / 140 tokens per frame, images 1 FPS / 280 tokens each. All methods share
  the same decoded frames and cached embeddings.
- Development positive references cover at least 50% of a selected event.
  Normal references do not overlap any annotated push/punch or kick.
- No validation/test labels construct the bank. No event times or actor boxes
  enter test scoring. An unlabeled mean of all windows in each recording is
  allowed for context centering: this is offline/transductive processing, not
  a demonstrated causal live-camera method.

Compared methods:

1. Existing fixed policy: image/video OR, positive similarity >=.35, margin >=.02.
2. Same prompt score, scalar threshold calibrated on validation.
3. Full-window references: top-three positive-reference cosine mean minus
   top-three normal-reference cosine mean.
4. Centered references: subtract each recording's mean video feature, normalize
   the residual, and apply the same reference comparison.
5. Fixed equal mean of prompt and centered-reference scores. Weights are not fit.

The primary screening metric is **event coverage**: at least one selected
candidate covers >=50% of an event's duration. Optimize for >=95% validation
coverage, then the fewest selected windows. Ties prefer the simpler method.
Write the chosen method and all thresholds to `lock.json` before test inference.

A second, more conservative protocol was declared while seq14 was still being
encoded, before test processing: leave one development recording out, build both
reference classes from the other eleven, and score the excluded recording.
Threshold caps from these development cross-fit scores were frozen in
`crossfit-caps.json`. The final threshold is the lower of that cap and the
validation-only threshold. Select the method using coverage in **both** cross-fit
and validation, then the fewest combined candidate units (simpler ties).
`crossfit-lock.json` records this choice without reading test scores. This extra
check prevents self-neighbor matches from giving an artificially optimistic
development result and reduces dependence on four validation recordings.
It still does not turn a small regression dataset into a population guarantee.

Also report temporal IoU >=.3 recall and an all-windows oracle ceiling. Very
brief events cannot achieve .3 IoU against a six-second window, even when the
event is fully inside it. Coverage must not be relabeled as precise temporal
localization or used to claim the original final event-detection targets pass.

Candidate count measures potential verification units; it is not measured Qwen
HTTP calls. Selected union duration removes overlap between windows. Non-target
window counts are not distinct false alarms/hour, and ignored actions are not
assumed normal.

## Implementation and reproduction

`evaluation/embeddinggemma2/continuous_screening.py` uses the existing localhost
broker and validates source SHA256, sampling configuration and model revision.
Each recording is cached atomically with frame/timestamp traces and telemetry;
an interrupted pre-lock run can reuse completed recordings. A locked run is
preserved rather than silently retuned.

Install the optional XLS reader separately from either inference environment:

```powershell
python -m pip install --target scratch/evaluation-tools xlrd==2.0.1
```

With `PYTHONPATH` containing `backend` and the repository root, branch services
running and no event job active:

```powershell
python -m evaluation.embeddinggemma2.continuous_screening
```

On a fresh checkout, `python -m evaluation.embeddinggemma2.continuous_screening --prepare`
downloads and verifies the two original archives and the label workbook without
starting inference. This run installed `xlrd` only into the ignored evaluation
tools directory; neither model environment was upgraded.

Outputs remain ignored under `scratch/event-retrieval/continuous-ut`.

Cross-fit reproduction, after development features exist:

```powershell
python -m evaluation.embeddinggemma2.crossfit_screening --lock
# Wait for the continuous run to finish, then evaluate the frozen choice:
python -m evaluation.embeddinggemma2.crossfit_screening --evaluate
python -m evaluation.embeddinggemma2.probe_continuous_transfer
```

The lock command waits at most ten minutes for the main validation lock; its
development caps are saved before waiting. The transfer probe reads existing
indexes and never changes application jobs, event reviews or thresholds.

The context experiment is motivated by visual normal references in
[NOVA](https://arxiv.org/abs/2609.06360), but it is **not** a reproduction of
NOVA's CLIP method or first-frame normality anchor. Our per-recording mean may
contain anomalies and is tested as an engineering hypothesis. Paper benchmark
scores do not transfer to this Gemma implementation.

## Results

20 original recordings, 1,379.1 seconds (~23 minutes), 470 windows. All source
hashes are distinct and every annotation lies within its original recording.
The bank contains 47 positive and 192 normal windows. Test consists of four
source recordings, 249.73 seconds, eight selected events and 85 windows.

| Screening policy | Validation events covered / candidates | Test events covered / candidates |
|---|---:|---:|
| Existing .35/.02 | 4/8; 10/102 | 4/8; 13/85 |
| Validation-only prompt | 8/8; 27/102 | 6/8; 23/85 |
| Validation-only window references (selected) | 8/8; 21/102 | 4/8; 16/85 |
| Validation-only centered references | 8/8; 30/102 | 7/8; 32/85 |
| Validation-only fusion | 8/8; 29/102 | 7/8; 30/85 |
| Cross-fit conservative prompt | 8/8; 62/102 | 8/8; 43/85 |
| Cross-fit conservative window references (selected) | 8/8; 27/102 | 5/8; 19/85 |
| Cross-fit centered references | 8/8; 30/102 | 7/8; 32/85 |
| Cross-fit fusion | 8/8; 29/102 | 7/8; 30/85 |

Both pre-test selection rules chose window references, and both failed the test
coverage target. **Reject their use as the sole hard filter.** In particular,
the conservative reference method selected zero windows in seq17, missing both
selected events. Its development cross-fit and validation result did not predict
this source shift.

The predeclared conservative prompt arm (margin `-0.017801880836486816`, similarity
floor .35) covers all eight test events. This deliberately admits ambiguous
positive-vs-normal similarities to second-stage review. It sends 43/85 windows,
reducing candidate units 49.4% relative to every-window verification, while
increasing them relative to the existing strict rule. It is a promising
high-recall **regression result**, not a claim that the pre-test winner passed
or that population recall is 100%. Eight positives give a wide uncertainty
interval (~67.6–100% Wilson, before recording dependence).

Full-window normal references fix the earlier catastrophic Normal634 behavior:
both reference thresholds select 0/150 windows instead of the earlier trimmed
bank's 150/150. They select 49/92 (validation-only) or 53/92 (conservative) on
Fighting011. These are transfer decisions, not annotated temporal recall.
Centering selects 72/150 Normal634 windows; fixed fusion selects 9/150. Reject
the idea that centering automatically removes domain errors.

The all-windows oracle covers 8/8 test events but reaches tIoU .3 for only 7/8.
The short-event/window-size limitation is therefore measurable, not merely a
hypothesis. Coverage does not satisfy the original final temporal acceptance gate.

### Actual Qwen confirmation and temporal zoom

After inspecting Gemma results, a diagnostic paired run compared the frozen
reference arm with the frozen conservative prompt control. Their union has
44 candidate windows (19 reference, 43 prompt, 18 shared). Common windows reuse
the same Qwen reply for a fair paired comparison. No labels enter requests.

The original eight-frame, six-second confirmation leaves only 3/8 covered test
events positive for the prompt arm (2/8 for references). Three units remain
unverified: Qwen claims an action from one frame, giving equal start/end indices,
so the broker correctly refuses to certify a transition. Do not coerce those
outputs into positives. Raw reasons also reject some annotated punches as
gestures or insufficiently visible contact; this supports investigation of
sampling and action definitions, not a proven single cause.

A post-test exploratory ablation uses up to eight freshly decoded frames over
the central three seconds of each same candidate. Thresholds, candidate IDs,
Qwen model and prompt remain fixed; no ground-truth time centers the crop. It
is not another blind test or a production policy change.

| Prompt arm confirmation | Events covered in actual verification interval | tIoU .3 matches | Unverified units |
|---|---:|---:|---:|
| Uniform six-second input | 3/8 | 3/8 | 3 |
| Central three-second input | 5/8 | 5/8 | 0 |
| Keep original positives; zoom only other units (paired-cache analysis) | 6/8 | 6/8 | 1 |

Central sampling alone loses one event that the original confirmed. The cascade
preserves prior positives and leaves unresolved evidence pending. Its 6/8 is
derived from the two measured reply sets, not an independently timed adaptive
execution. It still misses two punches and fails the >=85% final recall target.
It needs a fresh camera-held-out evaluation before selecting a deployment method.

`sampling-analysis.json` distinguishes a six-second candidate footprint from
the three-second interval actually shown to Qwen. Credit is not granted for a
true event outside the latter. Raw Qwen artifacts retain candidate-coverage
fields; the analysis contains the corrected verification-interval comparisons.
No final precision, false alarms/hour or clinical-readiness claim is made.

### Resources and checks

- Feature indexing: 1,009.10 summed seconds, excluding the initial text-model
  load (35.03 seconds), download and auxiliary cross-fit analysis.
- Gemma: reported peak allocated/reserved VRAM 1.83/2.02 GiB; maximum sampled
  post-request whole-card use 3.20 GiB and broker RSS 2.52 GiB. Whole-card use
  includes other programs and post-request samples can miss transient peaks.
- Qwen paired union runs: uniform 143.47 seconds / 52 private-worker HTTP
  requests; central 143.30 seconds / 54 requests. These cover 44 units, not all
  85 test windows. Internal JSON repair generations are excluded from counts.
- Qwen sampled whole-card peak 11.28 GiB. Process-tree peak RSS 7.63 GiB
  (uniform) / 7.61 GiB (central).
- Prompt-only subsets use 51/53 HTTP requests from those respective shared runs.
  The hypothetical adaptive cascade requires 91 requests according to cached
  per-unit counts: higher recall has a real compute cost. Cascade wall time was
  not measured. Do not infer 49.4% actual request savings from candidate reduction.
- 42 targeted tests passed: 16 continuous/cascade tests, seven prior reference
  tests and 19 runtime-worker tests. Checks include source exclusion, test-set
  calibration rejection, short-event metrics, missing-reply preservation and
  withholding credit for events outside a zoomed verification interval.

Decision: retain the existing application policy while these profiles remain
experimental. Use broad prompt candidates as the next high-recall prototype;
reference scores can assist prioritization, with uncertain candidates retained.
Temporal zoom with fallback merits further validation, but its current 75%
event coverage is insufficient. No `validation_passed` lock was installed, no
human review was changed and no merge into main was performed.

Reproduce the paired diagnostic after the frozen continuous run:

```powershell
python -m evaluation.embeddinggemma2.verify_continuous_candidates --compare
python -m evaluation.embeddinggemma2.verify_continuous_candidates --compare --sampling central
python -m evaluation.embeddinggemma2.analyse_continuous_sampling
```

Graft context was consulted first. The new evaluation modules are unindexed and
Graft CLI/MCP is unavailable; no graph rebuild is claimed.

For a separate rerun, pass the same fresh `--output <directory>` to every
experiment command. Existing locks and Qwen raw results are preserved rather
than overwritten. The original source archives are shared, not redownloaded.
