# Explicit temporal evidence ablation

Protocol recorded before inference, 8 October 2026. No new weight training.

The existing llama.cpp bridge submits chronological images followed by one text
prompt containing a frame/time table. Qwen3-VL's technical report describes
explicit text–timestamp alignment for video. This motivates testing text labels
immediately before each image. This bridge still uses multiple image inputs;
it does not claim to implement native video positional encoding.

Sources: [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631),
[official Qwen repository](https://github.com/QwenLM/Qwen3-VL).
[Free Video-LLM](https://arxiv.org/abs/2410.10441) also motivates investigating
temporal sampling and spatial regions without training. No performance result
from these papers is assumed to transfer to CapRL Q6 or this dataset.

Two paired arms use identical up-to-eight full-window frames on **every** window
of development recordings seq1 and seq2. No Gemma threshold or ground-truth time
selects windows in this verifier-only experiment. Existing official annotations
are used only to score the four selected target interactions and out-of-target
positive windows. Test recordings 17–20 are not read for this experiment.

- `legacy`: existing image sequence and verification prompt.
- `temporal-v1`: interleaved zero-based image index and absolute timestamp,
  explicit before/during/after comparison, two distinct evidence frames for a
  positive, and original query/semantic requirements retained during JSON repair.

This is a combined input/prompt/repair ablation, not isolation of each component.
Semantic action requirements, model, quantization, image pixels and generation
budgets are unchanged. No requirement is relaxed to fit acted dataset labels.
Missing temporal images fail instead of silently shifting frame indices.

Preregistered promotion rule: run validation only if selected-event coverage
strictly improves and out-of-target positive window count does not increase.
Otherwise retain the opt-in experiment and reject activation. Even a pass is
insufficient for the full acceptance gates. Defaults remain `legacy` throughout.

Run `python -m evaluation.embeddinggemma2.compare_temporal_verification` with
backend on PYTHONPATH. Raw measured runs are ignored under
`scratch/event-retrieval/temporal-dev/results.json`; existing outputs cannot be
overwritten. Brokers are restarted only after checking no jobs, no resident
model children, and that both processes belong to the experimental worktree.

## Measured development outcome

| Metric | Legacy | Temporal-v1 |
|---|---:|---:|
| Selected target events covered | 2/4 | 2/4 |
| Events matched at temporal IoU >= 0.3 | 2/4 | 2/4 |
| Positive windows outside target/ignored interactions | 1 | 0 |
| Positive windows | 4 | 3 |
| Unverified windows | 4 | 2 |
| Private worker HTTP requests | 44 | 43 |
| End-to-end verification run time | 106.91 s | 114.38 s |

Both arms submitted the same 41 windows. The removed out-of-target positive
was seq2, 33–39 s, overlapping the official hugging annotation at 36.33–39.83 s.
The original answer described the embrace as a forceful grab. Temporal-v1
preserved all three positive windows in seq1 and removed this extra seq2 result.
These window counts are not a statistically validated precision measurement.

Both missed the same two seq2 target interactions (punch at 14–16.43 s and push
at 21–23.37 s). Raw answers around those intervals describe arm extension or
brief contact but reject sufficient evidence of forceful motion. This supports
investigating spatial detail and temporal coverage; it does not establish that
resolution alone is the cause. No event requirement was relaxed after reading
these answers, and uncertain/invalid results remain unverified.

The preregistered promotion rule fails: recall did not strictly improve. Do not
run validation to search for a more favorable result. Keep the mode opt-in and
the application default unchanged. The arms were run once in a fixed order;
timings are descriptive and do not isolate warm-up/order effects. Two sources
and four target events cannot establish generalization.

59 relevant checks passed: 50 experiment/worker checks and nine existing
sequential-cascade/contract regressions. Tests cover exact image-label ordering,
unchanged legacy ordering, missing images, chronological alignment, event-aware
JSON repair, broker forwarding and unknown-mode rejection. Both branch workers
were restarted safely; main and user-generated output remain untouched.

## Next evidence-driven experiment

The previous dense-half validation experiment is still the measured recall
improvement (3/8 to 6/8), at substantially higher request cost. This new experiment
adds a useful false-positive diagnostic, not further recall improvement.

Prioritize a paired full-frame versus automatic person-region experiment on
development data, then freeze a policy before validation. Region proposals must
come from a pretrained detector or motion, never annotation boxes. Preserve both
people, surrounding interaction context and whole-frame fallback if localization
fails. Compare with the same temporal frames and token budget; record crops and
retained context for visual review. Combine with dense sampling only after its
separate contribution is measured. A positive spatial result must be tested on
normal hugs, handshakes and assisting someone, not just target actions.

The training-free spatial/temporal approach in Free Video-LLM motivates this
next experiment; it is not a replicated method or a guaranteed gain on our task.
If ambiguous contact remains unresolved, retain a review-needed state rather
than lowering confirmation requirements to inflate recall.

Graph refresh remains unavailable because the graft executable is absent; use
the existing main-checkout wiring cards for original source context and this
commit's diff for shifted definitions.
