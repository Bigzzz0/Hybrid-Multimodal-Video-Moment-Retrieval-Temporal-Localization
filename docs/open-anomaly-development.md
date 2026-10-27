# Open-category surveillance anomaly screening

## Implementation and limits

Experimental opt-in `open_anomaly` mode on the existing Gemma branch. It combines
scene-specific verified normal references with broad semantic prompts, followed
by contextual Qwen visual review. No model weights are trained. This is a proposed
combination inspired by existing research, not a reproduction or a claim to be
the best anomaly detector. Existing specific-event screening remains the default.

Profiles and immutable banks live in separate SQLite tables. Complete indexed
windows wholly inside an explicitly verified development interval are copied into
the bank; partial windows never borrow frames outside the normal interval. One
centroid per reference window prevents duplicate image slots dominating the three
nearest references. Raw channel scores are independently gated with an OR, not
averaged across modalities. Missing references permit text-only screening and are
marked `reference_missing`. Novelty measures deviation, not danger.

Jobs snapshot context, prompts, configuration, thresholds and reference version.
Adding a normal example publishes a new bank without changing old jobs. A source
or model/configuration mismatch requires reindex/rebuild; an old validation lock
cannot silently apply to a changed bank. Context is immutable within a profile;
create another profile for a changed camera/scene.

Selected windows are merged with a three-second gap, then split into units no
longer than twelve seconds. Qwen receives eight event frames and up to two before
and two after, five seconds of context on each side, plus at most two separately
labelled normal-reference images. Actual source-frame indices are deduplicated,
so very short/boundary clips can contain fewer than twelve frames. Timestamped
event frames and references are explicitly separate. Evidence frame indices do
not replace the original event boundaries.

Qwen returns `normal`, `suspicious` or `uncertain`, with observations, deviation,
alternative explanation and uncertainty. `uncertain` is represented with
`event_present: null` at the broker boundary. At most two three-second retries
use the highest screening-score spans. A normal retry cannot erase an uncertain
full unit. Errors/OOM remain unverified with raw evidence and reviews retained.
Qwen makes no audio confirmation, diagnosis, identity or intent inference.

## API/demo

1. Register/upload and index videos normally.
2. `POST /api/v1/events/profiles` with `name`, `context`.
3. `POST /profiles/{id}/references` with `ranges` containing `video_id`, `t_start`,
   `t_end`, `split: development`, `verified_normal: true`. This only copies ready
   index vectors; it does not run inference. `expected_version` detects stale edits.
4. `POST /jobs` with `kind: screen`, `mode: open_anomaly`, `profile_id`, `video_id`.
5. Inspect `/events` in the branch frontend. Review decisions remain separate from
   model assessments. Only an explicitly rejected, confirmed-normal development
   event may be added through `/profiles/{id}/normal-events/{event_id}`.

`GET /profiles`, `/profiles/{id}`, `/profiles/{id}/bank?version=...` expose metadata
without vector blobs. Local manual additions rely on the operator's development
declaration; dataset evaluation additionally checks manifests/hashes/official
splits before encoding. Never add a validation/test example to a reference bank.

Unvalidated defaults: per-channel novelty .25, semantic similarity .35 and margin
.03. They are experimental guesses, not calibrated probabilities. Runtime locks
under `normality-locks/{profile_id}.json` must match configuration, bank version,
policy, prompt hash and validation manifest. Benchmark selection never installs
these locks or promotes the frontend default automatically.

## Data and evaluation protocol

Primary sources:

- ShanghaiTech: https://svip-lab.github.io/dataset/campus_dataset.html
- UBnormal: https://github.com/lilygeorgescu/UBnormal (CC BY-NC-ND 4.0, synthetic)
- Label conversion: the authors' `scripts/write_frame_level_ground_truth.py`:
  union inclusive abnormal-track frame ranges, convert by actual video FPS.
- Research: https://github.com/Yuchen413/AnomalyRuler,
  https://github.com/lucazanella/lavad, https://arxiv.org/abs/2505.15205.

ShanghaiTech official OneDrive metadata currently returns HTTP 401 unauthenticated;
its public landing page is HTML, not accessible video/label files. No alternative
mirror is silently substituted and no ShanghaiTech result is claimed. Its official
training data are normal-only: they cannot supply anomalous validation examples.
Use an independent labelled validation source or collect a separate local
validation set; never repurpose official ShanghaiTech test clips for calibration.

UBnormal's official 15 GB archive supports byte ranges. `prepare_ubnormal` reads
the ZIP directory and downloads original selected MP4s, tracks and license only.
It does not fetch pixel masks, test split lists or test clips. The initial smoke
subset fixes scene 4 and lexicographic recording order before reading labels:
two normal training recordings, one normal validation, one abnormal validation.
This validates wiring, not generalisation, and is not a hospital benchmark.

Manifests require scene, recording, dataset and official split, source/license,
source SHA256, duration and timed labels. Hash/recording leakage is rejected.
Scenes supply independent development references; validation selects thresholds.
UBnormal simulated scenes and real video results are reported separately. Existing
UT/Kamakura checks remain regressions, not open-category accuracy evidence.

Four arms use the same features and recordings: normal references, text semantics,
their union, union plus contextual Qwen. Validation grids are fixed before outcomes
(novelty .05/.1/.2/.3/.5; margin 0/.03/.1; minimum similarity .35). Select recall
>=95% first, then higher recall, fewer unmatched events, less selected video time,
stable threshold tie-break. Freeze all arms before Qwen; never tune on test.

Event evaluation uses maximum one-to-one matching with temporal IoU >=.3. Merge
overlapping positive units first, so duplicates do not earn multiple matches.
Unknowns are counted separately and never treated as confirmed events. False
alerts/hour uses fully normal recordings only, not total mixed video duration.
Report sample counts, video-bootstrap 95% intervals, actual requests and unknown
request counts, indexing/verification/total times, broker PyTorch allocated/reserved
VRAM separately from whole-card sampled VRAM and Qwen process-tree RSS.

Targets remain screening recall >=95%, final recall >=85%, precision >=80%, <=2
false alerts/normal-video hour and >=50% Qwen request reduction. A hypothetical
all-window baseline is explicitly labelled unmeasured; don't claim measured speedup
unless both policies actually ran with identical verification prompts/budgets.
The four-recording smoke set cannot establish any production acceptance target.

## Reproduction

Run in the branch checkout with `PYTHONPATH=backend;.` and isolated inference Python:

```powershell
python -m evaluation.embeddinggemma2.prepare_ubnormal
python -m evaluation.embeddinggemma2.run_open_anomaly --manifest scratch/event-retrieval/datasets/ubnormal-open/manifest.json --output scratch/event-retrieval/open-anomaly-ubnormal-smoke-v1
python -m evaluation.embeddinggemma2.open_anomaly_eval --manifest scratch/event-retrieval/datasets/ubnormal-open/manifest.json
```

Use a new output path for another run; protocol/validation locks are preserved.
Raw databases, clips, labels, frames and results remain under ignored `scratch/`.
The evaluation CLI allows test manifests only with a validation lock whose digest
is explicitly bound in `development_lock_hash`. No test set is opened by the smoke
runner. `run_open_test` provides frozen-threshold inference on separately prepared
test manifests and a copy of the development reference database.

The `graft` CLI is unavailable in this environment. Existing graph cards were read
before source inspection; added modules are not indexed. Run `graft build` when
the CLI is available to refresh the local graph.


Completed measurements and limitations: [open-anomaly-results.md](open-anomaly-results.md).
`publish_open_demo --experiment <completed-run>` copies experimental evidence into
the local demo without installing thresholds. Known validation/test videos carry
split metadata and cannot be redeclared as development references.
`benchmark_open_baseline` actually measures all twelve-second verification units;
the six-second index-window count is a separate hypothetical baseline. Sampled
peaks aggregate across batches/retries; missing request counts stay unknown.
