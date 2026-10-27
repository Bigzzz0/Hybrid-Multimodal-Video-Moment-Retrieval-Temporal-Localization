# Observable motion and temporal coverage — 8 October 2026

## Question and scope

Does a motion-focused decision definition, denser temporal coverage, or both
recover observable person punch/push actions without increasing development
out-of-target windows? This is a verifier experiment, not proof of assault,
injury, intent, force magnitude, or calibrated probability. Keep contact
visibility separate from motion. Default application verification remains legacy.

Previous action-expert experiment retained 8/8 validation event-containing
six-second candidates, but existing Qwen evidence covered only 5/8 events.
Some negative reasons described directed arm movements yet rejected them for
unseen contact/force or inferred playfulness. Some motion-selected two-second
views missed the action timing. Both hypotheses are therefore worth separating.

## Fixed protocol

Qwen is the existing `caprl_qwen3vl_4b_q6` through the central localhost GPU broker.
No retraining, model substitution, action-score hint or annotation-dependent
sampling. Annotated timestamps/boxes never enter Qwen requests. Both prompt modes
use identical full-frame images and interleaved exact frame indices/timestamps.

Development: all 41 windows from seq1/2, four selected punch/push events,
including surrounding normal windows. Validation, only after a passing frozen
development choice: the existing 62 Gemma candidates from seq13–16, eight events.
No seq17–20 test access. These recordings were used in prior experiments; this
is internal validation, not a new blind test or the official UT cross-validation.

Two prompt modes:

- `temporal-v1`: unchanged strict person contact/force semantic requirements.
- `behavior-v2`: experimental observable directed punch/push motion; additionally
  asks `action_motion`, `contact_observed`, `receiver_response`, `uncertainty`.
  Two distinct evidence frames still required. Contact can be unclear without
  inventing an impact. Pointing, greeting, hugging, assistance and object-directed
  actions remain excluded. Motion `uncertain` stays unverified, not normal.
  Malformed/contradictory raw JSON is unverified even if the private worker's
  response parser supplies defaults. This mode supports only person punch/push.

Two sampling arms:

- `uniform`: original full-window eight-frame control.
- `grid`: retain the same full-window result, then retry nonpositive windows in
  chronological two-second clips with one-second stride (five for a six-second
  window), eight frames each; stop after a positive. Include a shorter source
  tail, never sample outside its bounds. Short windows may have a single retry.

These form four development arms. Uniform inference is shared with the grid arm,
not rerun or double-counted. Both modes interleave within batches of at most eight.
All arms disable caption generation explicitly; application default still creates
captions as before. Verification budget is 256 new tokens in both modes.
Private JSON repair is capped at 128 tokens for old mode, 256 for the longer new
schema. Therefore repair cost can differ and must not be described as exactly
matched token work. Broker request counts exclude private internal repair calls.

The protocol records implementation/control/source SHA256 before inference and
checks video SHA256 against the feature provenance. Stops if application jobs
are active. Writes partial artifacts after each batch and refuses to overwrite
existing runs. Unverified attempts remain present through retries.

## Selection and validation gate

Before viewing development outcomes, selection was fixed to:

1. Zero development out-of-target positive views under the existing evaluator.
2. Event coverage greater than matched `temporal-v1/uniform` and greater than the
   historical best motion-focused development result of 3/4.
3. Prefer higher coverage, fewer pending windows, then fewer private HTTP
   requests; final tie uses arm name for determinism.

If no arm passes, write a no-selection lock and stop before validation. Otherwise
freeze chosen arm + development artifact hash + implementation hashes, then
evaluate only it and the matched strict uniform control on validation. Do not
choose again using validation results, activate in production, or open test.

Metrics: event coverage >=50% target duration in at least one positive view,
secondary temporal IoU >=.3, out-of-target positive views, pending windows and
private requests. Out-of-target count excludes kicks and unselected punch/push
annotations as in prior work; it is not event precision or false alerts/hour.
Full windows and short retry views differ in length, so report coverage and IoU
together. Sum of arm requests is descriptive, not separate actual arm wall times.
Actual batch wall times/whole-card GPU/process-tree RSS are saved separately.

## Reproduction and artifacts

With branch workers loaded from this checkout and PYTHONPATH containing backend
and repository root:

```powershell
python -m evaluation.embeddinggemma2.behavior_factorial
# Run only if development selection-lock contains a passing selected_arm:
python -m evaluation.embeddinggemma2.behavior_factorial --split validation
```

Ignored output under `scratch/event-retrieval/behavior-factorial-dev` and
`behavior-factorial-validation`: protocol lock, results, sampled frames; development
also writes selection-lock. Measured runs cannot be silently overwritten.

Relevant source: `evaluation/embeddinggemma2/behavior_factorial.py`, broker
`worker.py`, opt-in contract and VLM prompt. Tests cover interval boundaries,
selection gating/cost ties, timestamp alignment, schema consistency, unclear
contact, unknown propagation, original-context repair and no-caption evaluation.

Graph cards from main were read before source scoping. New evaluation modules
remain unindexed; graft executable is unavailable, so graph refresh is pending.

## Measured outcome

Development completed on all 41 windows:

| Prompt / sampling | Event-containing view coverage | Out-of-target positive views | Pending source windows | Private HTTP requests |
|---|---:|---:|---:|---:|
| Strict temporal-v1 / uniform | 2/4 | 0 | 2 | 40 |
| Strict temporal-v1 / conditional grid | 4/4 | 0 | 4 | 212 |
| Observable behavior-v2 / uniform | 3/4 | 0 | 3 | 40 |
| Observable behavior-v2 / conditional grid | 4/4 | 2 | 1 | 189 |

The frozen gate selected **temporal-v1/grid**, not the relaxed semantic mode.
The two out-of-target positive views in behavior-v2/grid are overlapping views
of the same seq1 hug, not two independently false events. Manual inspection of
frames 0/4/5 of seq1 window12 stage1 showed both people opening their arms and
approaching; Qwen called frame4→5 a punch. No prompt was adjusted after this
observation. Full-window context and a short view can disagree.

Actual combined development wall time **1051.43 s**, **401** private HTTP requests
(212+189; uniform controls are already included in the grid arms). A one-frame
tail is unverified before requesting Qwen, so uniform has 40 calls for 41 windows.
No claim that denser verification is cheaper: selected grid uses 5.3× the matched
uniform private request count. Private internal JSON repairs remain excluded.

An additional **post-selection diagnostic** uses Qwen's claimed start/end frame
timestamps instead of the entire positive input view. Development coverage at
>=50% duration and temporal-IoU>=.3 is **0/4 in all four arms**: model-selected
frame pairs tend to span only .267–1.0 s, shorter than annotated full actions.
This does not erase a correctly observed action, but prevents claiming accurate
complete boundaries. The analysis separately reports whether actual cited frame
spans overlap the target at all. Neither diagnostic changed the frozen gate.
Do not substitute evidence indices for clip boundaries without further validation.

Validation completed under the frozen `temporal-v1/grid` selection:

| Validation arm | Event-containing view coverage | View IoU >=.3 | Out-of-target positive views | Pending source windows | Private HTTP requests |
|---|---:|---:|---:|---:|---:|
| Matched strict uniform | 3/8 | 3/8 | 0 | 4 | 62 |
| Selected strict conditional grid | 6/8 | 6/8 | 0 | 4 | 337 |

Actual selected cascade + its shared matched control took **1126.08 s** (18.77
minutes), including decoding, batching, repeated model loading/unloading and
artifact writes. Do not add the 62 control requests to 337: they are already
included in that cascade. Grid uses **5.44×** matched uniform HTTP requests.
These are not warm query p50/p95 or independently measured uniform/grid wall times.

New positive evidence recovers the seq16 punch missed by motion-focused sampling;
seq13 and seq14 punches remain missed. Historical dense-half verification achieved
the same 6/8 coverage with 188 private requests and 563.36 s; its legacy prompt
and caption policy differ, so it is a historical cost reference, not an isolated
sampling comparison. There is no evidence here to replace it with the more
expensive new grid as the preferred accurate option. No final >=85% recall gate.

Sampled whole-card GPU peak / Qwen process-tree RSS peak:

| Measured combined run | Whole-card GPU peak | Process-tree RSS peak |
|---|---:|---:|
| Development | 11.237 GiB | 6.946 GiB |
| Validation | 11.073 GiB | 7.839 GiB |

ResourceSampler samples every .5 s. Whole-card GPU includes other applications;
these are not per-Qwen PyTorch allocated/reserved bytes. No OOM in measured runs.

Validation Qwen **cited frame evidence** overlaps 6/8 events in the selected grid
(3/8 uniform), supporting actual observations rather than merely a positive
six-second window. However, interpreting those frame pairs as complete event
boundaries gives duration coverage **0/8** and IoU >=.3 **1/8** for grid (0/8
and 0/8 uniform). Evidence spans and complete action boundaries are different
outputs. Retain source-window context for playback, and do not claim accurate
event start/end localization from these indices.

**Decision: do not activate the relaxed prompt or five-retry grid by default.**
Keep the opt-in modes/benchmark and raw evidence for continued comparison.
Next justified experiment is a matched pretrained Qwen3-VL-4B-Instruct verifier
versus the caption-focused CapRL model, then separately native-video processing
versus multi-image input. Use the same development recordings, prompt and frames
first to isolate model effects; freeze choices before further evaluation. A
larger model or additional retries are not automatically better. Repeated use
of this validation set requires fresh labeled recordings before deployment claims.

The two latest retries added no new covered event in this validation run, but
retry-budget selection must be made from development in a new protocol, not
retroactively tuned on this validation curve. No test-set evaluation, default
switch or merge. **81 targeted tests pass**, with two existing dependency
deprecation warnings. Raw outputs and source recordings remain outside Git.

Final read-only analysis: `python -m evaluation.embeddinggemma2.analyse_behavior_factorial`,
ignored `scratch/event-retrieval/behavior-factorial-analysis.json`. It preserves
raw-run hashes and keeps view coverage, cited-frame overlap, and inferred frame
boundaries separate. Production event-time handling was not changed.
