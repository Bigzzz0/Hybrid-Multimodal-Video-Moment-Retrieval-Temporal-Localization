# Automatic person-region verification ablation

Protocol recorded before inference, 8 October 2026.

Test the hypothesis that spatial detail limits Qwen action verification.
Use all 41 development windows of seq1/2, including normal interactions. Reuse
the immutable temporal-v1 full-frame control; no validation/test labels select
frames, windows, detector settings or crops. Model, quantization, prompt, frame
timestamps/count and output budget match that control. Match the original image
canvas (720x480) using ordinary cubic resizing, not generative super-resolution.
No new weights, environment upgrade or GPU detector is required.

One fixed crop per whole window includes the union of all pretrained OpenCV HOG
person proposals across the eight frames, with 20% context on each side and at
least half the original width/height. Expand the rectangle to the input aspect
ratio. With fewer than two detection frames or a near-full-frame union, use the
original images. Save every proposal, crop rectangle, fallback, detector hash,
OpenCV version and preprocessing time. Includes all detected proposals, but cannot
guarantee inclusion of people the detector misses. Visually inspect sample crops.

The algorithm is motivated by training-free spatial/temporal selection in
[Free Video-LLM](https://arxiv.org/abs/2410.10441), not a reproduction or a
claim that its benchmark gains transfer here. Detector API and bundled pretrained
people detector: [OpenCV HOG documentation](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html).

Promotion rule: target coverage must improve strictly over the same temporal-v1
full-frame control without an increase in out-of-target positive windows. Only
then freeze the spatial method for validation. If it fails, reject activation;
do not tune crop thresholds on validation to reverse the decision. Cropping is
experimental and does not change application defaults or review decisions.

Reuse of a cached control makes time comparisons descriptive, not a randomized
simultaneous benchmark. Matching canvas/frame count is an approximate visual
budget control; the llama.cpp bridge does not return actual vision-token counts.
Coverage is not precision. Full-frame context and direct contact may still be
unobservable after cropping; such uncertainties require review, not invented
evidence. Existing label handling and interval coverage rules remain unchanged.

Run `python -m evaluation.embeddinggemma2.compare_person_region`. Immutable raw
output: ignored `scratch/event-retrieval/person-region-dev/results.json`.

Development-only detector feasibility revision, before SSD inference: HOG used
full-frame fallback for all 41 windows (37 too-wide, four insufficient). It
therefore provides no spatial treatment. Add a separately saved SSDLite320
MobileNetV3 COCO_V1 arm, on CPU FP32 with four threads and fixed person class=1,
score >=0.5. All crop/context/promotion rules remain unchanged. This is detector
selection on development, not validation tuning. Record exact weights hash and
installed torch/torchvision versions, and download official pretrained weights
to ignored experiment cache without installing/upgrading dependencies.

[Torchvision model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html).
Run `python -m evaluation.embeddinggemma2.compare_person_region --detector ssdlite`.
Separate immutable output: `scratch/event-retrieval/person-region-ssdlite-dev/results.json`.

## Measured development results

Both region arms cover 2/4 selected target events, the same as temporal-v1
full-frame control, with zero out-of-target positives. Neither passes promotion.
HOG actually submitted all original frames (41/41 fallbacks), so it is a detector
feasibility failure, not evidence that spatial cropping cannot help. SSDLite
cropped 28 windows and fell back on 13 (ten wide, three insufficient). A sampled
seq2 crop was visually checked and retained both interacting people.

| Run | Region preparation | Verification wall time | Private HTTP requests |
|---|---:|---:|---:|
| Full-frame control, prior saved run | none | 114.38 s | 43 |
| HOG | 11.88 s | 114.36 s | 43 |
| SSDLite | 14.03 s | 117.45 s | 43 |

SSDLite end-to-end time was 134.65 s including its cold initialization and
preparation; HOG was 126.46 s. Detector peak RSS was not sampled. Qwen sampled
whole-card GPU peak was 11.072 GiB for the SSD run, including other applications.
These cached-control/fixed-order times are descriptive. No validation run for
spatial cropping and no default activation. This tests this particular automatic
region policy, not every detector/crop strategy.
