# EmbeddingGemma 2 experiment

Branch: `codex/embeddinggemma2-experiment`.

Setup is separate from execution. The default command only checks prerequisites.
The benchmark commands only instantiate models and generate embeddings when
`-Run` is supplied.

The interactive UI is a separate entry point: models run when the user submits
a search, not when opening the page or starting the local servers.

```powershell
# Install isolated overrides and download both pinned model snapshots, no inference.
./scripts/embeddinggemma2.ps1 -Setup

# Check the local environment, weights and media metadata, no inference.
./scripts/embeddinggemma2.ps1

# Start the experiment only after the user requests execution.
./scripts/embeddinggemma2.ps1 -Run
```

## What is prepared

- SigLIP2 NaFlex baseline and EmbeddingGemma 2 candidate, pinned to commit revisions in `config.json`.
- A separate `.venv-embeddinggemma2` that uses CUDA PyTorch from `.venv-inference` through a `.pth` file. New Transformers and Sentence Transformers packages are installed in the experiment environment; the original environment is not upgraded.
- Identical 1 FPS samples for both encoders, 768-dimensional normalized vectors, raw Thai and English queries, and one warmup query per model.
- The existing smoke set: 2 real videos and 28 queries. No event description or ground-truth boundary is used to generate embeddings or choose samples.
- Sequential model loading and unloading to limit GPU memory consumption.
- Query embeddings use EmbeddingGemma's `SearchQuery` prompt. Images use no text prefix. The optional caption encoder uses the `Document` prompt.
- EmbeddingGemma uses BF16 on compatible CUDA hardware, otherwise FP32, and disables the unused audio encoder. It never uses FP16.
- A default prerequisite check that reads media metadata and local package/model files without model inference.

## Interactive UI

```powershell
./scripts/start-embeddinggemma2-ui.ps1
# Open http://127.0.0.1:3000/experiments/embeddinggemma2
```

Choose a benchmark video or upload MP4/MOV/AVI/WEBM/M4V (up to 1 GB), enter a
Thai or English query, and search one model or compare both. Click a retrieved
frame or suggested time interval to seek the shared video player. Progress,
actual query time, loading time, and indexing time are shown separately.
UI model preparation includes a text warmup so first-use GPU setup is not
presented as warm query latency.
The latest benchmark metrics can be expanded below the results.

The API listens only on loopback port 8012. The UI is part of the existing
frontend on port 3000. Completed benchmark frame vectors are reused only when
model revisions, media hash and sampling settings match. Uploaded videos and
sampled frames stay under `scratch/embeddinggemma2/ui`; their vectors are cached
in memory and rebuilt after an API restart. Models are loaded sequentially to
fit GPU memory, so comparing both may include model switching time.

API integration checks (deterministic fake encoders, no model inference):

```powershell
./.venv-embeddinggemma2/Scripts/python.exe -m pytest evaluation/embeddinggemma2/test_service.py -q --basetemp=scratch/embeddinggemma2/tests-ui
```

## Native video and audio mode

The UI now offers `Gemma · Video + Audio`. It loads the complete 744M-parameter
checkpoint, including the audio encoder. Every retrieval unit contains up to
six chronological frames sampled at about 1 FPS and the corresponding real
16 kHz mono audio. The video processor uses its maximum supported 1,120 soft
token budget per frame, includes relative timestamps, and disables truncation.
This leaves space within the 8,192-token context for audio and delimiters.
Text queries use `SearchQuery`; media inputs have no task prefix. Burned-in
subtitles are visible to the vision encoder. There are no synthetic captions,
chapter titles, annotation labels, or generated transcripts in corpus inputs.
Silent uploads explicitly use video only. Arbitrary uploads are supported;
first indexing of a long clip can take several minutes and shows progress.

The Kamakura run uses the user's local file, not a downloaded substitute:

```powershell
./.venv-embeddinggemma2/Scripts/python.exe -m evaluation.embeddinggemma2.kamakura --prepare
./.venv-embeddinggemma2/Scripts/python.exe -m evaluation.embeddinggemma2.kamakura --run --profile gemma_av
./.venv-embeddinggemma2/Scripts/python.exe -m evaluation.embeddinggemma2.kamakura --run --profile siglip
./.venv-embeddinggemma2/Scripts/python.exe -m evaluation.embeddinggemma2.kamakura --run --profile gemma_frames
```

Outputs are in `scratch/embeddinggemma2/kamakura`. The native AV corpus is
restored into the UI only for a video with a matching content hash and model
revision. Stop the experiment API during resource benchmarks to avoid competing
GPU jobs, then restart it with `scripts/start-embeddinggemma2-ui.ps1`.

All profiles rank the same 198 six-second windows. Visual controls encode frames
individually and average their normalized vectors within each window; the native
profile jointly encodes video and audio. Media and text warmup occur before
indexing measurements. Each query is measured five times. Model loading,
indexing and queries have separate resource traces. PyTorch allocated/reserved
memory, process RSS/CPU, device-wide GPU utilization, memory and power are recorded.
Device power is not an incremental energy measurement; short phases may have
few one-second telemetry samples. Decode/audio preparation is outside model
indexing time. Each profile runs in its own process, sequentially.

`kamakura_queries.json` freezes ten manually reviewed visual concepts, each
phrased in Thai and English, with positive window IDs from a visual review of
all 198 center frames before model rankings. `CenterFrameHit@1/@5` measure that
visual reference, not exact event boundaries, audio understanding, no-match
detection, or generalization. The ten paired concepts are not twenty independent
events. Review sheets and all individual predictions are preserved for auditing.

The checkpoint supports further task prompts, smaller MRL dimensions, and
fine-tuning; this mode does not claim to evaluate every possible configuration.

## Visual-only comparison and reports

The initial comparison isolates **visual frame retrieval**. Both sides use the same
RRF, timeline smoothing and temporal proposal extraction from this project.
It excludes caption fusion, query expansion, scene detection, calibration,
production candidate grouping, SAM and VLM verification. It is not a replacement
for the full production benchmark.

The model vectors, sample images and reports live only under
`scratch/embeddinggemma2/`. The experiment never writes vectors to the application's
LanceDB or changes production model settings. Each run gets its own output folder.

Reports include:

- FrameHit@1 and FrameHit@5: whether the top retrieved frames lie inside an annotated event.
- R@1 and R@5 at temporal IoU 0.3, 0.5 and 0.7; mean IoU and temporal F1.
- Thai and English metric breakdowns and predictions for every query.
- Model-loading time, frame-indexing time, warm query p50/p95 latency and peak allocated VRAM.
- Model revisions, package versions, dataset/video hashes, sampling settings and paired bootstrap comparisons.

The tiny smoke set is useful for finding integration problems and obvious
regressions. It cannot demonstrate general superiority or satisfy the project's
held-out acceptance criteria. The existing database has no indexed caption rows
under the benchmark IDs, so caption semantic search is not falsely scored using
the ground-truth event descriptions. Caption fusion can be evaluated separately
after a matching frozen caption corpus is prepared.

## Setup record

`scratch/embeddinggemma2/readiness.json` records prerequisite status and explicitly
states `inference_run: false` and `benchmark_run: false`. It is a setup report,
not a measurement of retrieval quality.

## References

- [Google model card](https://huggingface.co/google/embeddinggemma-2)
- [Google inference guide](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)

The Graft command is unavailable on this machine. Repository context was read from
the existing `graft/` cards before the referenced source spans. Once Graft is
installed, run `graft build` to index these experiment files.
