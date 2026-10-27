# Screening iteration log — 8 October 2026

Branch: `codex/embeddinggemma2-event-retrieval`. Experimental development only; no automatic merge into main.

## Round 1 — bounded temporal evidence

Problem: Fighting011 produced 86 candidate windows that were merged into two decisions. The second decision covered 21–274.67 seconds with only eight Qwen frames and failed caption validation.

Change: keep each verification unit at most six seconds, do not bridge absent candidate windows, and inspect the entire short window instead of narrowing it using the same unvalidated Gemma scores. Overlapping units remain separate candidate observations, not a count of distinct real incidents. Each screening run has its own event IDs; current results select the latest run per event type, including pending/interrupted results, while `include_history=true` retains earlier observations and human reviews.

Validation: 26 event/service/API tests passed, including a 90-window case, preservation of missing intervals, rejection of oversized input, and history/empty-result/interrupted-job behavior. Initial test setup failed because Windows denied access to the shared pytest temp folder; rerun with a fresh workspace-local temp directory succeeded. This is correctness evidence, not measured improvement in event precision or recall. Real-video evaluation follows after round 2.

Tradeoff: bounded units increase verification requests on a broad candidate region. Candidate thresholds remain unchanged and unvalidated. The previous long-clip results remain the baseline.

## Round 2 — independent action decisions and complete evidence

Change: Qwen verifies chronological action evidence before generating a description, without a caption hint. Event-specific requirements exclude resting/ordinary help and require an upright-to-ground transition for falls. Raw JSON must contain a real boolean, finite uncalibrated confidence, a reason, and valid frame indices; positive action decisions require at least two distinct frames. Partial JSON and clamped/default values from the private worker cannot become negative observations. Negative units retain their full decision reason without requesting an extra caption. Positive units and shouting visual context request a compact caption with one bounded retry (384 then 768 tokens), retaining both raw attempts. Caption failure does not erase a valid visual action decision; shouting still never receives acoustic confirmation from Qwen.

Frontend: selected evidence follows polling updates; results from an old video request cannot overwrite the current video's event list. Negative decisions and caption failures display their actual state instead of appearing to wait forever for a caption.

Validation: all 73 backend tests, 18 runtime/worker tests, and 8 frontend tests passed; production frontend build succeeded. Added tests cover malformed verdicts, invalid frame evidence, bounded caption retry, preservation of valid decisions after caption failure, and live selected-evidence updates. Real-video regression is recorded separately below.

Telemetry caveat: `qwen_calls` counts HTTP requests to the private worker, including broker retries. The private worker may perform additional JSON-repair generations internally, which are not counted. The theoretical `qwen_baseline_calls` is not a measured control and uses the previous caption-plus-verification strategy. Neither figure proves a model-generation reduction under the new prompts.

Reproduce on existing indexes with `python -m evaluation.embeddinggemma2.run_screening_regression`. The runner preserves raw jobs/events in an ignored JSON artifact and requests cancellation after a 30-minute watchdog per video. It does not alter human reviews, ground truth, thresholds, or index vectors.

Initial real-video checks: UT punch: one positive unit, zero unverified, two private-worker requests, 58.45 s including cold loading. UR fall: one positive and one negative unit, zero unverified, three requests, 9.54 s. Normal634 (448.66 s): no visual candidates, no Qwen requests, 0.18 s using existing vectors and cached prompts. These are model decisions on weakly labeled demonstration clips, not ground-truth accuracy. Full Fighting011 evaluation is ongoing at this checkpoint; raw output: `scratch/event-retrieval/iterations/round2-screening.json`.

Completed round-2 Fighting011: 86 Gemma candidate hits became 85 bounded units (overlapping short tail units can coalesce within the six-second limit). Qwen: 25 positive, 56 negative, four unverified; all 25 requested positive captions completed. Total screening 267.10 s, 112 private-worker HTTP requests, compared with the old 214.94 s / three requests / two overly broad decisions. More detailed verification cost about 24% more time; no claim of better measured precision/recall is made. Some reasons described pushing a gate rather than a person, motivating round 3. Raw baseline artifacts were preserved.

## Round 3 — target-specific decisions, honest telemetry, and review queue

Change: verification now explicitly targets contact with another person's body, excluding gates/walls/objects. Failed raw verification replies are retained for audit alongside caption attempts. The webpage separates positive/negative/undecided units and can filter the queue while leaving all results accessible; counts explicitly refer to overlapping observations, not distinct incidents. Human decisions remain separate.

Resource bug: empty/cached jobs could inherit the previous video's Qwen samples, and multi-batch jobs reported only the last batch's peak. Reset telemetry at job start and accumulate sampled peak RAM/whole-card VRAM, samples, batch count, and verification time across all batches. Resource values from the round-2 artifact must not be interpreted as whole-job peaks; Normal634's stale Qwen samples were not measurements for that video.

Validation: 28 targeted service/API tests and 19 worker tests passed, including stale-resource isolation, a two-batch maximum lower in the final batch, and retention of malformed raw verification output. Eight frontend tests and a production build passed after filter changes. Fresh real-video results follow below; the full raw output is `scratch/event-retrieval/iterations/round3-screening.json`.

Context graph: new Gemma/event code is not in the main checkout's existing Graft graph. Graft CLI/MCP is unavailable in this session, so a deterministic graph rebuild could not be performed. Main checkout and its uncommitted work remain untouched.

## Final round-3 real-video results

| Video | Positive units | Negative units | Unverified units | Worker HTTP requests | Screening seconds |
|---|---:|---:|---:|---:|---:|
| ut-punch-demo | 1 | 0 | 0 | 2 | 33.23 |
| ur-fall-demo | 1 | 1 | 0 | 3 | 13.88 |
| ucf-long-fighting011 | 22 | 61 | 2 | 109 | 295.74 |
| ucf-long-normal634 | 0 | 0 | 0 | 0 | 0.19 |

Fighting011 sampled whole-card peak VRAM: 11.40 GiB, including other applications. Qwen process-tree peak RSS: 5.59 GiB. Aggregate covers 11 batches and 510 samples (0.5-second interval). Peaks are sampled, not guaranteed absolute maxima. Gemma is unloaded during Qwen verification; these values are not Gemma allocated VRAM.

All 22 requested positive captions completed. Both unverified units retain the original raw verification reply; raw frame indices exceed the supplied frame budget. All 85 human reviews remain pending. Normal634 has no Qwen resource samples after the fix, matching its zero Qwen requests.

Comparison: round 2 produced 25 positive / 56 negative / 4 unverified units in 267.10 seconds; round 3 produced 22 / 61 / 2 in 295.74 seconds. Candidate hits stayed at 86 of 92 indexed windows. Prompt changes and inference variability can affect decisions. Fewer positive units or fewer format errors do not establish higher precision/recall; round 3 did not improve measured total screening latency. Person/object contact and forceful help/restraint still need manual temporal annotation and hard-negative validation.

Final software checks: 75 backend tests, 19 worker tests, 8 frontend tests (102 total), plus a successful production frontend build. Three implementation checkpoints were completed. Further accuracy tuning requires reviewed development/validation labels; thresholds remain experimental rather than being adjusted from this one long clip. No claim that the original recall/precision/false-alarm acceptance targets have passed.

## Training-free reference experiment, 8 October 2026

Compared five frozen-encoder methods using common cached image/video features,
official UT recording-disjoint development/validation/test and exploratory UR
development/validation previews. The method and thresholds were locked before
UT test inference. Video references won validation and retained all 8 UT test
positives with zero false positives among 12 normal clips; existing policy had
8 true positives and 2 false positives. Chronological signatures missed 3 test
positives; fixed fusion missed 1. UR test remains sealed.

Cross-domain transfer rejected activation: the locked UT video-reference method
selected all 150 windows of Normal634, whereas the existing policy selected
none. It still selected 86/92 Fighting011 windows; these are decisions, not
ground-truth temporal metrics. This small matched-domain success does not
validate universal event screening. No production policy or user review changed.

Seven new experiment tests and 19 worker tests passed. Full protocol, resources,
limitations and research sources: [training-free-screening.md](training-free-screening.md).
Raw ignored artifacts: `scratch/event-retrieval/training-free`.

## Continuous labeled regression and temporal zoom, 8 October 2026

Acquired the original UT recordings using their existing official interval
labels: 20 distinct sources, 23 minutes, 470 windows. Compared prompt margins,
full-window positive/normal references, per-recording centering and fixed fusion.
Both validation-only and leave-one-source-out conservative selection chose
references; these covered only 4/8 and 5/8 test events, so activation is rejected.
The conservative prompt control covers 8/8 (existing fixed policy 4/8) while
sending 43/85 test windows for verification. It remains a small source-disjoint
regression; earlier trimmed clips from these sources were already evaluated.

Actually ran Qwen on the 44-window union of frozen reference/prompt candidates.
For the broad prompt arm, eight-frame six-second input leaves 3/8 events
positive. A paired exploratory central-three-second input improves this to 5/8
with the same maximum eight-frame budget. Keeping original positives and using
zoom for other units covers 6/8 in cached-pair analysis, with higher request cost;
the adaptive cascade was not independently timed. Actual verification intervals,
not the original broader candidates, determine the zoom coverage credit.

42 targeted tests passed. No validation-passed lock was installed and no app
policy, human reviews, main checkout changes or merge were performed.
[Thai results](continuous-screening-results-th.md) and
[protocol, resources and full results](continuous-screening-experiment.md).
Ignored raw output: `scratch/event-retrieval/continuous-ut`.

## Adaptive dense Qwen validation, 8 October 2026

Implemented and actually executed conditional verification on the frozen broad
Gemma validation candidates (UT recordings 13–16, 62 units, eight selected
targets). Keep eight-frame full-window positives; otherwise inspect eight frames
in each chronological half, stopping at the first positive. No label enters
sampling or model requests; invalid evidence remains pending.

Positive-view coverage improved from 3/8 to 6/8, with zero positive views outside
target/ignored interactions under the existing label scoring. This is validation
coverage, not measured final precision. Requests increased from 66 to 188; summed
broker batch time increased from 163.54 to 464.89 seconds. The actual adaptive
run took 563.36 seconds end to end. Two punches remain missed; two candidates
remain pending. Keep as an experimental option, not a production default or
passed acceptance lock. 45 targeted tests passed.

[Protocol, costs and limitations](dense-verification-validation.md).
Ignored raw output: `scratch/event-retrieval/dense-validation/results.json`.

## Explicit temporal evidence, 8 October 2026

Researched Qwen's text–timestamp alignment and training-free spatial/temporal
selection. Added an opt-in temporal verification mode with per-image zero-based
labels/timestamps, explicit before/after evidence, missing-image rejection and
event-aware JSON repair. Legacy remains default. Actually compared identical
eight-frame inputs on every window of development seq1/2 (41 windows).

Target coverage remains 2/4 in both arms; out-of-target positive windows fall
from one to zero (a hug misclassified by legacy), and unverified windows from
four to two. Requests 44 versus 43, run time 106.91 versus 114.38 seconds.
The preregistered recall-improvement promotion rule fails, so no validation/test
run or default activation. 59 relevant tests pass. Next prioritized experiment:
automatic person-region detail with whole-frame context/fallback, then combine
with dense temporal sampling only if independently useful.

[Research, protocol and measured limitations](temporal-verification-ablation.md).
Ignored raw output: `scratch/event-retrieval/temporal-dev/results.json`.

## Spatial versus motion-focused temporal detail, 8 October 2026

Implemented CPU-only person-region preparation with fixed per-window padded
unions. HOG fell back for all 41 development windows. Pretrained SSDLite COCO
cropped 28/41 with no weight training or dependency upgrade; target coverage
remained 2/4 versus matched full-frame temporal-v1 control. Reject spatial-only
activation. Sampled crop retained both interacting people.

Implemented a label-free motion cue to select a two-second, eight-frame retry
after nonpositive Qwen, preserving original positives and uncertainty. Development
coverage improves 2/4 to 3/4, with no increase in out-of-target positive views.
Frozen algorithm/hash/candidates then evaluated on validation: 3/8 to 5/8, still
zero out-of-target positives under current interval scoring and two pending units.
Actually ran matched validation control and retries: 65+66 private requests;
phase wall times 174.52+245.72 seconds, not one combined end-to-end measurement.
Historical dense halves remain higher in coverage (6/8), with a different prompt
and greater request cost. Neither meets acceptance gates; no production activation.

62 targeted tests pass. Fixed adaptive-view interval-union/extent reporting for
unsorted and empty views; coverage and selection decisions unaffected. Preserved
raw artifacts and saved corrected analysis separately. Researched pretrained AVA
SlowFast action detection as the next model comparison; not installed or run yet.

[Spatial experiment](person-region-ablation.md),
[temporal results and next-model evidence](motion-zoom-ablation.md).
Ignored outputs: `scratch/event-retrieval/person-region-dev`,
`person-region-ssdlite-dev`, `motion-zoom-dev`, `motion-zoom-validation`,
and `spatial-temporal-analysis.json` under the same scratch root.

## Pretrained action evidence, 8 October 2026

Implemented isolated CPU FP32 SlowFast R50 AVA + pretrained SSDLite person ROIs,
32 unique frames in a label-free motion-selected two-second span. No retraining,
worker environment upgrade, GPU competition or production default change.
Actually ran all 41 development windows and 62 frozen Gemma validation candidates.
Strict threshold gives 7/8 validation candidate-window coverage; development-only
exploratory recall-first threshold gives 8/8 in 16/62 windows. Actual two-second
span coverage is 7/8. Neither candidate score is a final event confirmation.

Validation CPU wall time 87.23 s, sampled process RSS peak 1.430 GiB; 11 unverified
windows remain unknown. Offline matching of existing Qwen temporal controls and
motion retries still covers only 5/8 events, both before and after filtering to
the 16 candidates. No new Qwen calls or measured end-to-end savings claimed.
Prioritize temporal confirmation/uncertainty review next; do not promote small
candidate-recall results into production precision/recall claims.

[Protocol, measurements and remaining bottleneck](action-expert-ablation.md).
Ignored outputs: `action-expert-dev`, `action-expert-validation`,
`action-expert-recall-lock.json`, `action-expert-analysis.json`,
`action-expert-cached-cascade.json` under `scratch/event-retrieval`.

## Frozen prompt/sampling factorial, 8 October 2026

Implemented opt-in behavior-v2 with separate action/contact/reaction/uncertainty
evidence and strict raw-schema consistency. Ambiguous actions remain unverified.
Added evaluation-only caption suppression, retaining the production default.
Actually compared unchanged temporal-v1 versus behavior-v2 on identical full
frames and conditional overlapping two-second/one-second-step eight-frame retries.

Development all41 windows: strict uniform2/4, strict grid4/4, behavior uniform3/4,
behavior grid4/4 with two out-of-target views of one hug. Strict grid selected
under the prewritten zero-extra/recall gate and frozen before validation.
Validation matched strict control3/8 to grid6/8, zero out-of-target positive views,
four pending source windows,62 versus337 private HTTP requests. Actual validation
cascade1126.08 s; whole-card GPU peak11.073 GiB, Qwen process-tree RSS7.839 GiB.
Historical dense halves already achieved6/8 with188 requests/different prompt.
Reject default activation: no accuracy advantage over that reference and higher cost.

Added read-only cited-frame diagnostics: evidence overlaps6/8 target events,
but treating frame pairs as full event boundaries gives0/8 duration coverage and
1/8 IoU matches. Do not substitute evidence pairs for clip boundaries.81 tests
pass. No model retraining, test access, production default switch or merge.
Prioritize a matched base Qwen3-VL-4B-Instruct comparison before more retries.

[Protocol, complete results and decision](behavior-factorial-ablation.md).
Ignored raw runs/locks/frames: `behavior-factorial-dev`,
`behavior-factorial-validation`, `behavior-factorial-analysis.json` under
`scratch/event-retrieval`.


## Open-category normality screening, 8 October 2026

Added immutable per-scene normal-reference versions, independent novelty/semantic
gates, contextual Qwen normal/suspicious/uncertain decisions, bounded retries,
API/profile/review UI and frozen continuous-video evaluation tooling. Explicitly
verified development ranges only; known validation/test sources cannot be relabelled
into a bank. Models remain frozen and the existing default remains unchanged.

Fetched an official UBnormal scene-4 subset via ZIP byte ranges: two normal development
clips and normal/abnormal validation clips, one labelled abnormal interval.
Normal-reference and combined arms cover1/1; text-only covers0/1. Contextual Qwen
retains1/1, no extra event on15.03 seconds normal exposure. Not general accuracy.
Actually ran matched twelve-second all-unit Qwen baseline: four HTTP requests vs
two screened requests (50% reduction on this subset). Final cached-index rerun
44.48 s, Qwen phase28.25 s; sampled whole-card GPU11.51 GiB, Qwen RSS6.56 GiB.
Do not confuse cold indexing, cached inference, PyTorch memory or whole-card peaks.

ShanghaiTech official access returns401; no primary-dataset or held-out-test result
claimed. Smoke results never install production thresholds. Published two experimental
evidence units and the development-only bank into the local review UI.
125 backend tests and two frontend tests pass, with TypeScript checking successful.
Preserved the initial failed metrics run (sub-microsecond rounded tail), then fixed
and tested duration tolerance. Graph refresh unavailable: graft CLI absent.

[Implementation/protocol](open-anomaly-development.md), [results](open-anomaly-results.md).

## Open anomaly: multi-scene validation and matched-context ablation

Expanded the official UBnormal synthetic subset to scenes 3/4/5: six normal development
recordings and ten validation recordings, seven temporal events and 60.13 s normal exposure.
Pinned official split-list revision, archived pre-label selection and verified cached video
CRC/size against the official ZIP. Official test remains unopened.

Stopped the first expanded run after finding that whole-database index reuse could carry
old Qwen events/profiles/reviews. Replaced it with a whitelist copy of index data for manifest
videos only, filtered predictions by job ID and tested read-only source preservation.
Metrics now reject absent recording predictions; stage diagnostics separate screening,
verification and boundary matching losses. Baseline binds the manifest and job's profile version.

Four-arm results: normal-only6/7+3FP, text-only4/7+1FP, combined6/7+3FP,
combined+Qwen4/7+0FP. Actually ran all-unit Qwen:4/7+0FP,19requests versus15screened
(21.1% reduction). Screened Qwen batches169.50s versus146.13s baseline; cross-video batching
differs, so no latency improvement claimed. Preserved raw/failed runs; did not replace defaults.

Second round changed only Qwen scene_context on every one of the same19units, keeping
frames/timestamps/normal references unchanged. A camera-normality policy recovers two events:
6/7+1FP,recall85.7%,precision85.7%,144.11s. The extra normal false alert refers to a red outfit
and allegedly unnatural movement. No result was removed to meet a target. This is an all-unit
validation ablation, not an activated Gemma+Qwen pipeline or independent test result.
Frame/code hashes remain stable during inference. Model weights and demo defaults stay unchanged.

[Expanded results and reproduction](open-anomaly-multiscene-results.md).
