# Frozen Gemma: reference-based screening experiment

8 October 2026, branch `codex/embeddinggemma2-event-retrieval`.

## Protocol

No encoder, adapter, classifier or prompt weights are trained. Reviewed dataset
classes supply a development reference library. Validation selects scalar
thresholds and the method; this is calibration and data preparation, not weight
training. It is also not a data-free or completely zero-shot method.

Five alternatives share the same BF16 Gemma revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`, 768 dimensions, 6-second windows,
3-second stride, 2 FPS video at 140 tokens/frame and 1 FPS images at 280 tokens.
The original SearchQuery descriptions and .35 similarity floor remain fixed.

1. Original image/video OR prompt margin, calibrated on validation.
2. Mean video vector compared with the top three action references minus top
   three normal references.
3. Three chronological image bins, normalized then concatenated, compared with
   action and normal reference sequences. This is a coarse order signature,
   not an optical-flow or learned action model.
4. Equal mean of the video and chronological-reference margins.
5. Distance from normal references alone. Novelty can suggest a candidate;
   it cannot establish a specific behavior or intent.

The fixed existing policy (.35/.02) is reported separately. Each alternative
chooses the threshold with recall >=95% on validation, then fewer false
positives. Method ties prefer the simpler alternative in the listed order.
Every method can satisfy validation recall by selecting all clips: false
positives and held-out results are therefore essential.

UT-Interaction contributes 100 officially trimmed clips: punch/push are
positive; handshake/hug/point are normal lookalikes. Kicking is excluded rather
than mislabeled as normal. Source recording groups remain disjoint: 60
development, 20 validation and 20 test clips. Development has 24 positive and
36 normal references; validation and test each have 8 positive and 12 normal
clips. All thresholds are locked before test inference. Results are **clip
classification**, not continuous temporal localization or alarms/hour.

UR preview contributes 36 development and 12 validation clips for exploratory
fall screening. Its test remains sealed because temporal annotation review is
pending and participant identities across splits have not been audited. Preview
resolution is 320x240. These results are not a formal fall acceptance test.

Source hashes are checked before reading features. Cache keys include model,
sampling configuration and decoded MP4 duration. The manifest occasionally
retains original AVI duration after conversion; actual MP4 duration determines
sampling, and differences are reported. No silent missing-frame padding is added.

Implementation: `evaluation/embeddinggemma2/training_free.py`. Run from the
branch checkout with its existing inference environment and `PYTHONPATH`
containing `backend` and the repository root:

```powershell
python -m evaluation.embeddinggemma2.training_free
```

Requires the branch API/broker to be running and no screening job active. Outputs
and decoded frames remain ignored under `scratch/event-retrieval/training-free`.
The application policy and existing event reviews are not changed by this test.

## Research basis and limits

[Google's multimodal guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
supports within-modality similarity and video embeddings with controlled sampling
and vision budgets. It does not validate any of our event thresholds.

[NOVA](https://arxiv.org/abs/2609.06360) motivates visual normal references for
training-free anomaly detection with CLIP. This experiment uses explicit
development references rather than assuming initial test frames are normal.
It is not a NOVA replication, and CLIP paper results do not establish Gemma
accuracy.

[UT-Interaction](https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html)
provides the official action labels used here. No Qwen verdicts serve as ground
truth. Hospital scenes, shouting, longer continuous footage and unseen camera
domains still require independent evaluation before deployment.

## Measured results

The completed run encoded 148 clips: 100 UT and 48 UR development/validation
previews. UT thresholds and method were written to `lock.json` before processing
the 20 test clips. No UR test clips were opened.

| Method | UT validation TP / FP / FN | UT test TP / FP / FN | Test recall | Test precision |
|---|---:|---:|---:|---:|
| Existing fixed .35/.02 | 7 / 4 / 1 | 8 / 2 / 0 | 100% | 80% |
| Calibrated prompt margin | 8 / 9 / 0 | 8 / 8 / 0 | 100% | 50% |
| Video references | 8 / 0 / 0 | 8 / 0 / 0 | 100% | 100% |
| Chronological image references | 8 / 0 / 0 | 5 / 0 / 3 | 62.5% | 100% |
| Equal video + image-order fusion | 8 / 0 / 0 | 7 / 0 / 1 | 87.5% | 100% |
| Normal-only novelty | 8 / 3 / 0 | 8 / 4 / 0 | 100% | 66.7% |

Validation selected video references, threshold `0.008477270603179932`.
This selection was not changed after test results. The extra image-order
signature does not justify its cost here: it misses three positives on test,
and fusion misses one. A lower prompt margin restores validation recall by
admitting many normal clips; it is not an accuracy improvement.

Eight positive test clips are a small sample. The 95% Wilson interval for the
observed 8/8 recall is approximately 67.6–100%, before considering dependence
between clips from the same recording. It does not establish population recall
>=95% or clinical suitability.

UR **validation only**: existing policy TP=1, FP=1, FN=5; video references
TP=6, FP=0, FN=0. Threshold `0.04451143741607666`. The chronological and fusion
arms tie this validation result; video wins the predeclared simplicity tie.
No formal held-out fall result is claimed.

### Transfer to the existing long videos

A read-only probe applied the locked thresholds to each already-indexed window.
Reference features are aggregates of trimmed development clips, while the probe
features describe six-second windows. This is an exploratory transfer check,
not the same clip-classification test or a temporal acceptance evaluation.

| Video | Windows | Existing policy selected | Video references selected |
|---|---:|---:|---:|
| Fighting011 | 92 | 86 | 86 |
| Normal634 | 150 | 0 | 150 |
| UR fall demonstration | 2 | 2 | 2 |

The reference method fails on Normal634: every window is selected. Its top-three
action-reference similarities range 0.764–0.817 and normal-reference similarities
0.751–0.796; margins 0.0101–0.0207 all exceed the UT-selected threshold. These
relative similarities are not probabilities. The normal clip's street/camera
context differs from the UT greeting examples. Domain mismatch is a plausible
explanation, not proof of which internal feature Gemma uses. Counts alone do
not establish Fighting011 recall; its temporal annotations remain pending.

**Decision:** video references are the strongest training-free candidate for a
matched scene/domain, but this bank and threshold are rejected for general
application activation. The existing webpage policy remains active. Do not
replace the current policy using the impressive small UT test result alone.

The next useful experiment needs real normal and action references representative
of the target camera, plus scene-held-out validation. An explicit out-of-domain
check can keep uncertain clips on the existing Gemma/Qwen path rather than
asserting they are safe or abnormal. Keep visual fall transitions and audio
shouting evidence separate. This is a proposed next experiment, not a measured
or implemented accuracy guarantee.

### Resources and software validation

The successful feature-building pass took 334.42 seconds summed across 148
clips, including decode and broker calls; this excludes the failed initial
manifest-duration attempt. The UT reference video bank itself occupies 184,320
bytes (~0.176 MiB). A local 1,000-repeat CPU similarity benchmark measured p50
0.155 ms and p95 0.208 ms, excluding video decoding, embedding and HTTP. This
is not end-to-end query latency.

The 580 completed media responses reported maximum post-request process RSS
2.47 GiB and sampled whole-card use 3.85 GiB. Broker-reported PyTorch peak
allocated/reserved memory was 1.83/2.27 GiB. Those PyTorch peaks accumulate
since model load, which preceded this run; whole-card samples include other
applications and may miss transient peaks. No Qwen inference was requested.

26 targeted tests passed: seven new experiment tests and 19 existing runtime
worker tests. The real-data run also exercised source fingerprints, actual
container duration, model revision checks, common feature caching and a locked
held-out test. No production event policy or human review was modified.

Artifacts (ignored): `scratch/event-retrieval/training-free/{lock,results,transfer,diagnostics}.json`.
Transfer reproduction: `python -m evaluation.embeddinggemma2.probe_reference_transfer`.
The graph was consulted before inspecting these new unindexed modules; the
Graft CLI/MCP is unavailable, so no graph rebuild is claimed.
