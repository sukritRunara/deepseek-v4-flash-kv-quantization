# Engineering Report — DeepSeek-V4-Flash KV-Cache Quantization, Local Phase

*Written 2026-07-17, at tag `dgx-phase-complete-v1`. Audience: an engineer joining the
project (or returning to it) who wants to understand what was built and why, without
reading five worklogs. Deep references: `docs/V4_CACHE_ARCHITECTURE.md` (architecture),
`docs/DECISIONS.md` (D-001…D-010), `docs/WORKLOG.md` (per-task detail).*

---

## 1. The problem, in one paragraph

DeepSeek-V4-Flash is a 1M-context model. At long contexts, the memory that dominates a
serving GPU is not the weights — it's the **KV cache**, the per-token state the model
keeps so it doesn't recompute attention over the whole history for every new token. This
project asks: how small can V4's cache get (FP8? FP4? mixed precision per channel group?)
before output quality degrades, and what does that save in real bytes? The economics
forced a two-phase plan: a local development phase on an ASUS GX10 (ARM, one GB10 GPU,
**no model weights allowed** — the checkpoint is ~700 GB), building everything that
doesn't need the real model, followed by a paid phase on 4× RTX PRO 6000 GPUs (RunPod)
that runs the real checkpoint. This report covers the completed local phase.

The central constraint that shaped everything: **on a tiny randomly-initialized model you
can prove machinery correct, but you cannot learn anything about the real model's
quality.** So the local phase optimizes for *provable correctness* — bitwise gates, not
tolerances, wherever physically possible — so that the expensive phase spends zero time
debugging plumbing.

## 2. What DeepSeek-V4's cache actually is (the discovery)

The first task was pure source archaeology, because V4's architecture was undocumented in
this codebase and *not* what the predecessor project assumed. Key facts, all verified by
running code, not reading comments:

- **V4 is not MLA** (the V2/V3 latent-attention design the reference repo targets). There
  is no latent vector, no `kv_b_proj` reconstruction. Instead, V4 uses shared-KV
  multi-query attention: each token contributes **one 512-dim vector that serves as both
  the key and the value** (we verified K and V literally share storage). Layout:
  channels 0–447 are "content" (no position encoding), channels 448–511 carry RoPE
  (rotary position encoding). Because the value doubles as the key, the attention output
  gets an *inverse* rotation applied afterward — a clever trick that keeps position
  information relative.

- **Three layer types share one cache design.** Every one of the 43 layers keeps a
  **sliding window** of the last 128 tokens (uncompressed). On top of that:
  - *CSA layers* (~20 of them) also compress every 4 consecutive tokens into 1 pooled
    "compressed KV" entry, and use a small learned **indexer** that picks, per query, the
    top-512 most relevant compressed entries — so attention over a million tokens touches
    only 128 window slots + 512 selected compressed slots.
  - *HCA layers* (~20) compress every 128 tokens into 1 entry and attend to *all* of
    them (dense, but 128× fewer entries).
  - 3 layers are window-only.
  The compressors keep small bounded buffers for partially-filled windows — these never
  grow and are never worth quantizing.

- **DeepSeek already quantizes this cache** — that's the crucial finding. The official
  inference code (which we can read but not run without weights) applies, at the moment
  of writing to cache: FP8 (e4m3 format) on the 448 content channels in groups of 64 with
  power-of-two scales, keeps the 64 RoPE channels in BF16, and puts the indexer's state
  through a Hadamard rotation + FP4 in groups of 32 (applied to both keys *and* queries).
  The model was **trained with this quantization in the loop** (QAT), so this exact
  recipe should be near-lossless. The Hugging Face implementation — the only one we can
  actually run — has *none* of this: it's pure BF16. That gap defines the whole project:
  reproduce the official recipe in the runnable implementation, verify it, then explore
  beyond it (FP4 on the main cache, per-group mixed precision).

## 3. What was built — five layers, each gated by tests

Everything below lives in `src/v4_kv_quant/` with tests in `tests/` (90 total, all
passing) and CLIs in `tools/`. The single most important architectural decision (D-003):
**all quantization is injected by subclassing the cache-layer classes**, because every
cache write in the HF implementation flows through exactly three methods (`update()` for
the window, `update_compressor_states()` for compressed/indexer entries,
`store_compression_weights()` for buffers). We never touch the attention code or the
generated modeling files. Experiments are selected purely by which cache object you pass
in.

### 3.1 Baseline semantics (Task 01)

A tiny random V4 model (~0.26M params, all three layer types, shrunken windows so
boundaries are reachable in dozens of tokens) plus 27 deterministic tests that pin how
the *unmodified* cache behaves: prefill vs token-by-token decode equivalence, chunked
prefill, window rollover, compression exactly at window boundaries, entry counts, the
[content | RoPE] channel layout (proven at runtime, not assumed), and determinism.
These tests are the safety net everything later is measured against.

One genuine discovery came out of this: **the indexer's top-k selection is
tie-unstable.** Computing the same scores through two mathematically equivalent paths
(batched prefill vs one-token-at-a-time) produces ~1e-7 floating-point differences;
when two candidate entries are nearly tied, the top-k *set* flips and downstream logits
legitimately diverge. Consequence: indexer quality must be measured by **top-k overlap**
(how often the same entries get picked), never by logit closeness. This later became the
calibration metric for the indexer (D-004).

### 3.2 Stage B — QDQ simulation (Task 02)

"QDQ" = quantize-then-dequantize: round the values as the target format would, but keep
storing them in BF16. This measures pure *quality* impact with zero memory saving — and
the tooling proves the zero explicitly rather than asserting it.

Built: exact pure-PyTorch reimplementations of the official kernels' numerics (FP8 e4m3
with power-of-two round-up scales computed bit-exactly via `frexp`; FP4 e2m1 with
round-to-nearest-even ties, done in software because torch can't cast to its FP4 dtype
yet; an exact Walsh–Hadamard transform), a serializable policy object with five named
presets, cache subclasses applying the policy at write time, and a teacher-forced
comparison harness (baseline and quantized runs consume identical token histories, so
differences are attributable to the cache policy alone) with metrics: max/mean/RMS logit
error, KL divergence, ΔNLL, top-1 agreement, indexer top-k overlap.

Two traps worth recording: cache subclasses **inherit a `_layer_type` attribute that
auto-registers them over the stock classes globally** — without `_layer_type = None`,
defining our subclass would silently hijack every baseline run in the process
(test-pinned now). And the indexer stores its entries in the Hadamard-*rotated* basis, so
queries must be rotated symmetrically — done by wrapping the scorer module, the only
place that sees post-RoPE queries.

Acceptance gates, all bitwise: policy off ⇒ identical to stock cache; RoPE slice never
touched; each value quantized exactly once; chunking-invariant.

### 3.3 Calibration plumbing (Task 03)

The question "which parts of the cache tolerate how much quantization?" needs machinery:
enumerate **targets** (layer × state × contiguous 64-channel group for main KV; whole
vector per layer for the indexer, since rotation mixes its channels), perturb one target
at a time, measure damage, rank, and emit a **precision map** — a versioned JSON saying
which group gets which format.

The elegant part: a precision map with a single entry *is* the perturbation experiment,
and a full map *is* the deployable mixed-precision policy — so one consumer
(`MappedQDQCache`) serves both, and a full-coverage map is verified bitwise-identical to
the Task-02 policy cache. A smoke calibration on the tiny model runs the whole pipeline
end to end (activation stats → 16-target sweep → map → held-out evaluation) and produced
physically sensible results (the indexer ranked most sensitive; FP4 error ~4× FP8 error)
— but remember, random weights: this validates the pipeline, not the rankings.

### 3.4 Stage C — actual low-precision storage (Task 04)

Now the cache *really* stores small numbers: native `float8_e4m3fn` code tensors plus
1-byte power-of-two scale tensors for main KV (RoPE slice stored raw), and FP4 packed
two-codes-per-byte for the indexer. Dequantization happens on read, in pure PyTorch.

The load-bearing correctness gate (D-009): a hard contract `load(store(x)) == qdq(x)`
**bitwise**, verified up to model level — the storage cache produces logits and indexer
picks bit-identical to the Stage-B simulation cache. That means every quality number ever
measured in simulation transfers to real storage with no re-validation.

Measured on identical 112-token streams (tiny fp32 model): baseline 24,576 bytes, QDQ
simulation 24,576 bytes (**exactly 1.000× — "simulation saves nothing" is now proven
output, not a claim**), actual storage 10,758 bytes (**0.438×**), with scales, RoPE
slices, and buffers all itemized in the accounting. Two honest caveats baked into the
tooling: the fp32 tiny model overstates savings (the BF16 real model gives ~2× on
quantized slices, not 4×), and the pure-PyTorch read path is *slower* than baseline until
kernel fusion — which is deliberately future target-hardware work, so no timings were
recorded. Bonus finding: the stock sliding-window cache layer stores V as a duplicate
copy of K and retains hidden history through tensor views; our layers fix both.

### 3.5 Benchmark harness + RunPod tooling (Task 05)

One config-driven command structure that runs the tiny model locally today and the full
checkpoint on RunPod by swapping a JSON file: warmed-up, CUDA-synchronized trials
measuring TTFT, prefill/decode throughput, inter-token latency, per-GPU peak memory,
actual cache bytes, and quantize/dequantize micro-overhead — medians over trials,
per-trial data always saved, every output labeled non-transferable.

Plus the **landing test**: a source-only check battery driven by an expectations file
(platform, GPU identity, compute capability, FP8/FP4 dtype support, vendor checkouts at
pinned SHAs, no weights present, the Stage-C bitwise gate, optionally the full test
suite). It passes on the GX10 with GX10 expectations and will hard-fail a mis-provisioned
RunPod pod *before* four GPUs are rented. Launch scripts are guarded (refuse on
non-x86_64; weight download additionally requires an explicit env-var opt-in, a pinned
revision, and a free-disk check).

This task surfaced the sharpest environment gotcha of the phase: **this torch build
routes CUDA `torch.bmm` through a Triton-JIT kernel**, and Triton compiles its launcher
stubs against `Python.h` — absent on the GX10 (no `python3.12-dev`). V4's output
projection uses `bmm`, so *no* CUDA forward of the model runs locally. All local work
runs on CPU (which covers every correctness gate); the landing test enforces dev headers
on RunPod pods so the problem cannot recur where it matters (D-010).

## 4. What the supplied reference PoC contributed

The project started from an attached predecessor repo (`reference/v2_mla_poc/`, kept
read-only) — a KV-quantization proof of concept for the older DeepSeek-V2/V3 MLA
architecture. It ended up serving as a **methodology donor**: its ideas shaped the
experiment design, while every line of V4-facing code was written fresh, because V4's
architecture shares nothing with what it patches. The full per-component classification
is `docs/REFERENCE_PORT_MAP.md`; the short version:

**Reused (with modification):**
- **Teacher-forced comparison** — its trick of feeding baseline and quantized runs the
  *identical* token history (so logit differences are attributable to the cache policy
  alone) became our harness. The single most valuable inheritance.
- **QDQ function skeletons** — the round-through-a-grid-and-back structure of its FP8/FP4
  simulators carried over; the scale scheme was replaced to match V4's official policy
  (contiguous groups of 64/32 with power-of-two round-up scales, vs its per-channel
  absmax).
- **Calibration workflow shape** — activation stats via hooks, rank targets, serialize a
  precision config; our `build_map_from_sweep` (fp8/fp4 fractions) directly mirrors its
  `make_mixed_precision_config`, re-keyed to the V4 target taxonomy.
- **C4 calibration-data pattern** (seeded, non-overlapping token windows, saved token
  ids) — earmarked for the RunPod port.

**Not portable, by necessity:** its attention monkey-patch and cache class key off MLA
modules (`kv_a_proj_with_mqa`, `kv_b_proj`, `kv_lora_rank`) that do not exist in V4; its
sensitivity formula depends on `kv_b_proj` column norms and therefore has no V4
counterpart; its memory reporting counted theoretical bytes for values still stored in
BF16 (exactly the simulation-vs-storage confusion our Stage B/C split eliminates). Its
setup/download scripts were never executed (project ground rule), and its audited defects
(malformed `requirements.txt` line, references to files absent from the archive) were
documented and steered around rather than inherited.

## 5. Where things stand and what's next

**Done and tagged** (`dgx-phase-complete-v1`, pushed to GitHub): all six planned local
phases, 90 tests green (also from a clean checkout — verified via git worktree),
dependencies frozen, results manifest with hashes, calibration fixtures committed, no
weight bytes ever on this machine.

**Deliberately left for RunPod** (because they need the real model):
1. a real-corpus calibration loader (C4-style streaming; the tiny-model smoke tool shows
   the exact sequence to mirror) and a full-model calibration CLI;
2. real sensitivity rankings and the final precision map;
3. all quality claims (teacher-forced logit/KL/NLL/top-k-overlap on the checkpoint);
4. all performance claims and the final benchmark matrix at several context lengths;
5. Stage-D fused kernels (dequant-inside-attention, selective dequant of indexer-picked
   entries — the big long-context win, since decode reads ≤512 compressed entries
   regardless of context length).

The step-by-step continuation guide is `RUNPOD_START_HERE.md`.

## 6. Repo map (30 seconds)

| Path | What it is |
|---|---|
| `src/v4_kv_quant/qdq.py`, `storage.py` | numerical primitives: QDQ simulation + real storage (bitwise-matched pair) |
| `src/v4_kv_quant/qdq_cache.py`, `mapped_cache.py`, `storage_cache.py` | the three cache families: policy-QDQ, per-group-map-QDQ, actual-storage |
| `src/v4_kv_quant/policy.py`, `precision_map.py`, `targets.py` | what to quantize, serialized |
| `src/v4_kv_quant/harness.py`, `metrics.py`, `sensitivity.py`, `stats.py` | teacher-forced comparison, quality metrics, calibration sweep |
| `src/v4_kv_quant/bench.py`, `landing.py`, `memory.py` | benchmark engine, environment checks, honest byte accounting |
| `tools/` | CLIs over all of the above (each is `--help`-documented) |
| `tests/` | 90 tests: the acceptance gates for every task |
| `configs/` | benchmark configs, pod expectations, pinned source SHAs |
| `docs/` | architecture map, decisions D-001…D-010, worklog, reproducibility, this report |
| `prompts/01…05` | the bounded task specs each phase was executed against |
| `reference/v2_mla_poc/` | read-only predecessor PoC (V2/V3 MLA — ideas ported, code not) |

## 7. Principles that held up (recommended to keep)

1. **Bitwise gates over tolerances** wherever the comparison is exact by construction
   (identity policies, storage-vs-simulation, chunking invariance). Every real bug this
   phase was caught by an exact gate or an exact-invariant test.
2. **Inject at the narrowest owned boundary** — cache-layer subclasses, never model code.
3. **Prove the negative too** — the benchmark shows QDQ saves 0 bytes; the landing test
   proves no weights are present; the worktree run proves no uncommitted-file deps.
4. **Label non-transferability at the source** — every tool prints it and writes it into
   its JSON, so a number can't drift into a claim later.
5. **Separate what the tiny model can prove (mechanism) from what it can't (quality).**
