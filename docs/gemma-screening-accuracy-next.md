# Gemma screening accuracy: evidence and next experiments

8 October 2026. Branch `codex/embeddinggemma2-event-retrieval`.

## What was actually tested

An isolated prompt-bank comparison reused the same active 768-dimensional image/video vectors, model revision, SearchQuery prefix, .35 similarity threshold, .02 margin, and modality-OR rule. Only the negative text descriptions changed. The experiment adds people standing/sitting close, already lying down, pointing, gently supporting someone, striking a gate/door rather than a person, deliberate sitting/kneeling/resting, and bending then standing again. No application screening configuration, index, event, or human review was changed.

| Cached video | Old push/punch hits | New push/punch hits | Old fall hits | New fall hits |
|---|---:|---:|---:|---:|
| UT punch demo | 1 | 1 | 0 | 0 |
| UR fall demo | 2 | 0 | 2 | 0 |
| UR normal demo | 0 | 0 | 0 | 0 |
| Fighting011 (92 windows) | 86 | 37 | 0 | 0 |
| Normal634 (150 windows) | 0 | 0 | 0 | 0 |

The new bank removes all candidate windows from the actual fall demonstration. Therefore it is rejected for activation, despite reducing Fighting011 push/punch candidates by about 57%. It retains 20 of the 22 units marked positive by Qwen in round 3; that is agreement between models, not ground-truth recall. Qwen-positive units 96–102 and 228–234 seconds are excluded. This comparison does not establish precision or recall, and five clips do not validate generalization.

Raw result: `scratch/event-retrieval/gemma-precision/prompt-ablation.json`. Reproduce with `PYTHONPATH` including `backend`, then run `python -m evaluation.embeddinggemma2.probe_screening_prompts`; each new run writes a separate ignored JSON file. The read-only database probe rejects index/model revision and dimension mismatches and refuses to start when screening jobs are active.

## Why the simple approach fails

The current rule compares the best matching positive description with the best matching negative description. Adding more negative descriptions increases the opportunity for a negative to win. A genuine fall clip also contains ordinary standing and a person already lying down; suppressing those appearances can suppress the transition itself. This is an engineering interpretation of the failed experiment, not a demonstrated causal model of Gemma's internals.

Google's guide explicitly warns that similarity can remain high even for unrelated text and distinguishes task-specific prompts. Those text examples do not define a universal image/video threshold. Our existing local video scores already show that .35 is too low to discriminate scenes in Fighting011. Google recommends Clustering for arbitrary text categories rather than blindly assuming Classification works better. Neither text-only recommendation proves a better prefix for our video event task; compare prompts experimentally while keeping media inputs unprefixed.

Source: [Google EmbeddingGemma 2 inference guide](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers).

## Recommended development order

1. **Create reviewed temporal labels before choosing new parameters.** Mark action onset/end, normal lookalikes, and ambiguous/occluded intervals in development videos. Include helping someone stand, people already on the floor, ordinary greetings, and forceful contact with objects. Keep source videos separated across development/validation/test. Do not use Qwen verdicts as authoritative training labels or annotate unreviewed intervals as negative merely because no event was proposed.

2. **Fit a small supervised screening head over frozen Gemma features.** Start with per-event logistic regression using video embeddings as a low-cost baseline; compare image/video features and their learned weights rather than accepting either modality whenever one score passes. Keep Gemma weights frozen and keep the existing search backend available. Train on reviewed development labels, including the hard negatives. This is our proposed transfer approach, not an off-the-shelf classifier proven for Gemma.

3. **If the linear baseline confuses transitions, train a small temporal head.** Feed chronological frame/window embeddings into a small temporal convolution or adapter, with explicit masks for short tails. It must distinguish upright → falling → lying from lying → lying, and approach/contact/displacement from pointing or gentle support. Evaluate on held-out videos. VadCLIP provides research support for learning temporal adaptation on a frozen encoder, but its published model uses CLIP and trained adapters/prompts; do not reuse its weights or benchmark numbers as Gemma results.

   Source: [VadCLIP, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/download/28423/28826), [official code](https://github.com/nwpu-zxr/VadCLIP).

4. **Choose screening thresholds per event and modality on validation.** Optimize candidate event recall first, then minimize false alarms and Qwen load within the planned targets. Record the prompt/head revision with the threshold lock. Raising a global threshold until this clip looks clean can hide true events. A fall and a shout should not share a universal similarity cutoff; audio should be evaluated on actual acoustic labels separately.

5. **Only then test media budget/crops as controlled factors.** Compare 140 versus 280/560 vision tokens on exactly the same frames and labels. For distant people, experiment with person crops plus the full scene, without replacing the scene or inferring identities. Crop detection/tracking and extra encodes have their own cost and failure modes. Keep the current 768 dimensions while selecting the accuracy configuration; evaluate compression afterward. Higher resolution/budget does not by itself establish better recall.

   Source for configurable vision budgets and native video input: [Google multimodal guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers).

## Decision

The strongest next direction is reviewed hard-negative data plus a small classifier/temporal head on frozen Gemma embeddings, compared against the current prompt-based baseline. It is a reasoned recommendation requiring an experiment, not a claim that it is already the most accurate configuration. The failed prompt-bank result means adding negatives alone should not be deployed. Production screening remains unchanged by this research iteration; event recall, precision, temporal IoU, false alarms/hour, and resources still require the planned validation protocol.
