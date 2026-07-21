# Worklog

## 2026-07-20 (Future work: FP4 ladder Runs A+B — results in, hard stop for owner)

Owner-approved continuation (FUTURE_WORK.md): long-context eval plumbing built
(retrieval harness `src/v4_kv_quant/retrieval.py`; harness `logits_to_cpu`
offload; chunkwise-GPU metrics via `compute_device`; vectorized indexer overlap —
suite 107), four ladder rungs (fp4 0.2–0.8) built from committed sweep data, and
two eval passes run:

- Run A (2k/8k grown held-out + 2×32k): mid-rung dNLL differences sit inside
  per-sequence scatter (±0.003–0.005) — NLL cannot rank rungs at 32k power;
  overlap decays cleanly with FP4 fraction. No hard gate failures.
- Run B (2×65k NLL + retrieval 8k/32k/65k): at 65k the dNLL ordering becomes
  monotone (ladder20 +0.001 ≈ noise; 40/60/80 +0.004–0.005; all-FP4 +0.0066);
  **retrieval is a perfect 1.000 for every variant including all-FP4** —
  verbatim long-range recall survives whole-cache FP4 (real result), and the
  task saturates as a discriminator (instrument result). Official FP4-indexer
  overlap continues its slide: 0.871 @65k. The ratified map itself reaches
  0.906 @65k with zero dNLL — the absolute 0.9 overlap gate looks length-naive
  (see FUTURE_WORK "LADDER RESULTS" for the full table and owner options).

One tool bug fixed mid-run (new stages missing from the model-load gate).
Route-2 (gradient-weighted ordering) NOT triggered: monotone at 65k power.
**No new map ratified — stopped for owner review per plan.**

## 2026-07-20 (Owner ratifies the B4 precision map)

Owner ratified the D-015 provisional selection: `moderate` — all main-KV non-RoPE
FP8 e4m3, RoPE BF16, indexer BF16 — is now the project's calibrated precision map.
D-015 status and PROJECT_STATUS updated; no code changes.

## 2026-07-20 (GCP G4 — B7: final benchmark matrix, all variants — PHASE B COMPLETE)

### Goal

Plan step B7: full same-node matrix (`scripts/runpod/launch_4gpu_bench.sh` →
landing test, then `configs/bench_runpod_4gpu.json`: baseline/qdq/storage ×
1024/8192/65536, batch 1, 128 decode tokens, 5 trials + 2 warmup,
prefill_chunk 2048, policy `reference_official_qdq`, NATIVE P2P per D-014).

### Results (medians of 5; `results/benchmark_runpod_20260720T083335Z.json`,
copy at `artifacts/phase_b_gcp/benchmark_matrix.json`)

| prompt | variant | TTFT | prefill tok/s | decode tok/s | ITL p50 | cache | vs baseline |
|---|---|---|---|---|---|---|---|
| 1024 | baseline | 491 ms | 2086 | 5.0 | 195 ms | 13.4 MiB | — |
| 1024 | qdq | 531 ms | 1927 | 4.5 | 213 ms | 13.4 MiB | 1.000× (sim) |
| 1024 | storage | 544 ms | 1884 | 4.4 | 221 ms | 7.2 MiB | **0.535×** |
| 8192 | baseline | 5.45 s | 1503 | 5.0 | 196 ms | 60.5 MiB | — |
| 8192 | qdq | 5.73 s | 1429 | 4.6 | 212 ms | 60.5 MiB | 1.000× |
| 8192 | storage | 5.77 s | 1419 | 4.4 | 219 ms | 31.2 MiB | **0.516×** |
| 65536 | baseline | 85.1 s | 770 | 5.0 | 195 ms | 436.7 MiB | — |
| 65536 | qdq | 87.1 s | 752 | 4.6 | 212 ms | 436.7 MiB | 1.000× |
| 65536 | storage | 87.4 s | 750 | 4.4 | 218 ms | 223.1 MiB | **0.511×** |

- Peak alloc ~42/48/69 GiB (hottest GPU), dominated by weights; identical across
  variants at each length.
- **Actual storage halves cache bytes (0.51–0.54×, scales included)** at a cost of
  ~9–12% ITL and 1–6% prefill (unfused pure-PyTorch dequant-on-read — Stage-D
  fusion is the known remedy). QDQ sim saves nothing by design (labeled).
- **Native-P2P node vs RunPod host-staged (historical, NOT comparable, but worth
  noting):** baseline decode 5.0 vs 3.3 tok/s, ITL p50 195 vs 296 ms, TTFT
  491 ms vs 546 ms @ 1k. The D-011 caveat does not apply to these numbers (D-014).
- Micro-overheads @65k shapes: fp8 window-step encode 78 µs (RunPod: 146);
  fp4 indexer entry encode 219 µs (370); decodes 22–65 µs.
- Allocator OOM-retry warnings at 65k cells (same benign pattern as B2); all 9
  cells completed.
- Note: qdq/storage variants use the OFFICIAL policy (per config). A benchmark
  variant for the selected B4 map (indexer BF16 → slightly larger cache, no FP4
  indexer encode cost) is a candidate follow-up; quality-side comparison already
  done in B4 heldout/32k.

### Phase B status

B0–B7 all complete. Deliverables: bitwise-validated Stage-B/C machinery on the
real checkpoint (B3, GCP re-validated), calibrated provisional precision map
(B4/B5, D-015 — owner ratification pending), storage memory proof + full latency
matrix on a healthy-P2P node (B6/B7). Manifest regenerated (51 files).

## 2026-07-20 (GCP G4 — B4 overnight: refine/fp8spot/indexer8k, candidate maps, heldout; two methodology findings)

### Results

- **refine** (105 group-64 FP4 targets in top-15 states): scores 1.27–2.33e-2, no
  elbow; single groups score ≈ their whole state. **fp8spot** (8 worst states,
  FP8): 1.48–2.09e-2 — *the same* as FP4 on the same states.
  **Interpretation:** the KL score saturates at the selective-indexer flip floor —
  any perturbation of an early layer, however fine, cascades near-tie top-k flips
  through the ~40 downstream indexer layers (late layers score ~10× lower: short
  cascade). Rankings order states honestly; format choice (FP8 vs FP4) is NOT
  measurable by probe KL in early layers (D-004 regime). ΔNLL noise-level
  throughout; no NaN/Inf anywhere.
- **indexer8k sweep is a measurement artifact — do not use per-layer.** All 21
  targets showed uniform overlap ≈ 0.615–0.62 and dNLL ≈ +0.03, even layer 42 (no
  downstream cascade). Cause (confirmed in code, `mapped_cache.indexer_query_context`
  docstring): the indexer-query QDQ wrapper applies to ALL layers uniformly — a
  one-layer key perturbation scored 20 layers with FP4-rotated queries against BF16
  unrotated keys (mismatched bases → scrambled picks everywhere). Valid indexer
  evidence is the ALL-layers aggregate below. Per-layer indexer granularity is
  unmeasurable without per-layer scorer wrappers (extends D-008's "on/off per layer"
  to a practical "on/off for all layers" tonight).
- **Candidate maps** (D-015 composition: 105 refine group-records + 69 state-level
  screening records; indexer excluded by the 0.9 gate on the artifact data — see
  above; kept BF16 in all candidates): conservative fp8 0.75 (130 entries),
  moderate fp8 1.0 (174), aggressive fp8 0.85 + fp4 0.15 (173).
- **Heldout (2k + 8k, chunked, official + 3 candidates)** — KL / dNLL / top1 /
  8k-idx-overlap:
  official      1.19e-2 / +9.6e-4 / 0.9558 | 1.14e-2 / +8.4e-4 / 0.9485 / **0.952**
  conservative  1.13e-2 / −7.0e-4 / 0.9551 | 1.14e-2 / −1.5e-3 / 0.9485 / 0.970
  moderate      1.20e-2 / +9.2e-4 / 0.9591 | 1.19e-2 / +8.7e-4 / 0.9491 / 0.969
  aggressive    1.34e-2 / +1.3e-3 / 0.9496 | 1.39e-2 / −9.9e-4 / 0.9490 / 0.969
  Guardrails (D-015): aggressive FAILS (KL +12–21%); conservative and moderate pass;
  **official itself passes the 0.9 overlap gate in aggregate (0.952)** — the
  all-FP4 indexer is benign on held-out data at 8k, unlike what the per-layer
  artifact suggested.
- Decision now binary: official (= all-main-FP8 + indexer-FP4) vs moderate
  (= all-main-FP8 + indexer-BF16), statistically tied at 2k/8k. Discriminator:
  32k spot-check (selection pressure 8192 entries vs top-512).
- **32k spot-check (1 × 32768 held-out seq, disjoint stream region):**
  official  KL 1.521e-2, top1 0.9767, dNLL −2.9e-4, idx_overlap **0.886** (< 0.9!)
  moderate  KL 1.542e-2, top1 0.9771, dNLL −1.6e-4, idx_overlap **0.911**
  The official FP4 indexer's overlap decays with context (0.952 @ 8k → 0.886 @
  32k) and crosses below the D-012 gate; moderate holds. Other metrics tied — the
  overlap gate is the binding criterion (D-004: judge the indexer by top-k
  overlap, not logit closeness).
- **PROVISIONAL SELECTION (D-015): `moderate` — all main-KV non-RoPE FP8 e4m3
  (state-level for the 69 non-refined states, group-64 within the top-15), RoPE
  dims BF16, indexer BF16.** Staged as `precision_map.json`. Owner notes for
  ratification: (a) official shows NO NLL/top-1 damage even at 32k — if indexer
  memory savings outweigh the overlap gate, official is defensible; (b) a
  long-context retrieval eval would discriminate better than C4 NLL (CLAUDE.md
  metric list) — candidate follow-up.

## 2026-07-20 (GCP G4 — B4 run 1 complete: screening rankings + a vacuous-indexer finding; overnight delegation D-015)

### Goal

B4 run 1 rerun (`--stages corpus,stats,screening`) with the chunked stats stage;
then owner-delegated continuation (D-015 — owner "Go", map provisional).

### Results (run 1)

- corpus: reused `token_ids.json`; stats: 40/40 sequences with `--stats-prefill-chunk
  2048` (fix verified at scale); screening: 105/105 targets on probe [4, 2048],
  no NaN/Inf anywhere. Outputs in `results/calibration_full/`, copies in
  `artifacts/phase_b_gcp/calibration_run1/`.
- **Main-KV rankings** (FP4 whole-state perturbation, score = KL + |ΔNLL|): early
  layers dominate — top: layer2/compressed 1.98e-2, layer4/compressed 1.86e-2,
  layer9/window 1.80e-2, then layers 0–8 window/compressed ~1.5–1.8e-2. Tail (late
  layers) ~1.1–2.3e-3. Smooth ~8–17× range, **no sharp elbow**; ΔNLL is noise-level
  throughout (some negative), KL carries the signal. top1 agreement ≥ 0.95 even for
  worst single-state FP4.
- **Indexer finding: the 2k probe cannot measure indexer sensitivity.** All 21
  indexer targets scored exactly 0 with mean/min overlap = 1.0 over 171,780
  positions. Cause (verified in config): `index_topk = 512` ≥ compressed entries at
  2048 tokens (CSA ratio 4 → ≤512 entries), so top-k selection NEVER binds at the
  probe length — overlap is 1.0 by construction and the perturbation is invisible.
  Consequence: `build_map_from_sweep`'s overlap gate would pass every layer on
  meaningless evidence. Fix: new `indexer8k` stage (probe [2, 8192] → ~2048 entries
  ≫ 512; chunked prefill).
- **Map-composition gap found**: the committed map stage consumed ONLY refine
  records (top-15 sensitive states) + indexer, leaving the ~69 least-sensitive
  states without entries → BF16. Inverted for a deployable map; fixed per D-015
  (refine group-64 records + state-level screening records for non-refined states +
  indexer8k records).

### Changes (all unit-tested; suite 100 passed)

- `harness.run_teacher_forced(prefill_chunk=...)` + same-shapes-both-sides rule
  (chunk-kernel ulp noise flips selective-indexer picks — D-004);
  `run_sensitivity_sweep`/`measure_target` pass-through.
- `run_calibration_full.py`: `indexer8k` stage; map composition per D-015 +
  `--map-suffix` candidates; heldout evaluates official + every
  `precision_map*.json` (chunked, + held-out indexer overlap); `heldout32k` stage
  (disjoint 32k sample, D-012); config-only load for map-only invocations.

### Next step

Overnight: refine + fp8spot + indexer8k (one load) → 3 candidate maps → heldout →
guardrail selection (D-015) → 32k spot → B7 matrix.

## 2026-07-20 (GCP G4 — B4 run 1 attempt 1: corpus OK, stats stage OOM at 8k → chunked-prefill fix)

### Goal

B4 run 1 (`tools/run_calibration_full.py --stages corpus,stats,screening`), first
at-scale execution of the calibration pipeline.

### Findings

1. **Corpus stage works on the first real run of the `datasets` streaming path**
   (C4-en + codeparrot-clean, unauthenticated): 40 calib + 10 held-out sequences
   tokenized and saved → `results/calibration_full/token_ids.json`.
2. **stats_stage OOMs on the first 8192-token sequence** (all 32 × 2k sequences pass):
   it ran one-shot forwards, and eager attention asked for a 20 GiB score tensor on
   GPU 0 (`combined_logits`, modeling_deepseek_v4.py:739) — the same failure class as
   WORKLOG B2 (one-shot 65k OOM → bench engine grew `prefill_chunk`). The stats stage
   had simply never run at scale (RunPod was sealed before B4). Fix: chunked prefill
   through the same cache in `stats_stage` (`--stats-prefill-chunk`, default 2048,
   mirroring the bench engine).
3. **Chunked-vs-one-shot stats equivalence is provable only on the dense-indexer
   fixture** (new `test_stats_chunked_prefill_equals_one_shot`): write structure and
   element counts match exactly on any fixture; values agree to ~1e-6 relative. With a
   *selective* indexer, chunk-shaped-kernel fp dust (~1e-7) flips near-tied top-k picks
   and downstream layers legitimately diverge (measured 2.6e-1 on tiny layer2/window_kv
   with index_topk=2) — the exact D-004 mechanism, NOT a chunking bug. Methodological
   note for B4: aggregate stats over 40 sequences are insensitive to near-tie flips,
   and D-012 rankings come from the perturbation sweep, whose baseline and perturbed
   runs share identical shapes (self-consistent). Suite: 99 passed.

### Next step

Relaunch `--stages corpus,stats,screening` (corpus reuses saved token ids).

## 2026-07-20 (GCP G4 — step 5 re-validation: B1 GO, B3 ALL GATES PASSED — bring-up complete)

### Goal

`docs/GCP_TRANSITION.md` step 5: one-model-load re-validations on the GCP instance —
(a) B1 generation sanity, (b) B3 bitwise gates, both on native P2P (D-014).

### Commands run

```bash
ls /home/sukrit/models/DeepSeek-V4-Flash/model-*.safetensors | xargs -P 16 -I{} cat {} > /dev/null
.venv/bin/python <B1 scriptlet, RUNPOD_START_HERE step 1 w/ GCP path>   # artifacts/phase_b_gcp/b1_generation.log
.venv/bin/python tools/b3_identity_gates.py                             # artifacts/phase_b_gcp/b3_gates.{log,json}
```

### Results

- **B1: GO.** Coherent greedy continuation, word-for-word identical to the RunPod B1
  text ("…a technique that splits a model across multiple devices…"). Per-GPU
  allocated 34.3/40.0/40.0/31.0 GiB — same sharding as RunPod. Load **16 s** with warm
  page cache (RunPod: 868 s off MooseFS; local-PD + pre-warm is dramatically faster);
  generate 32 tokens in 99 s incl. first-call Triton autotune (not a perf number).
  Ran with NO P2P workaround → first real-model validation of native P2P (D-014).
- **B3: ALL GATES PASSED** via the committed driver. Gate A (QDQCache(baseline_bf16)
  == stock, logits AND indexer picks) and gate B (storage == qdq under
  reference_official_qdq) bitwise on both prompts. Informative official-QDQ vs
  baseline max|dlogit|: 15.31 natural / **4.844 random-512 — exactly the RunPod
  value** (same seeded ids), a strong cross-host reproducibility check. (Natural-prompt
  value differs from RunPod's 7.5 because the original scratchpad's natural text was
  not preserved; the committed driver embeds its own fixed text. Expected per D-004 —
  near-tie indexer top-k flips dominate; judged properly in B5.)
- Stage-B/Stage-C machinery fully validated on this host. **RunPod volume retention
  condition met (GCP_TRANSITION step 5) — operator may release the RunPod volume.**

### Next step

B4 run 1: `tools/run_calibration_full.py --stages corpus,stats,screening`
(first real exercise of the `datasets` streaming path), then owner reviews rankings
(D-012 stop point).

## 2026-07-20 (GCP G4 — bring-up steps 1–4: environment, P2P verdict, weights)

### Goal

Execute `docs/GCP_TRANSITION.md` steps 1–4 on the GCP instance
(`deepseek-v4-flash-g4-4gpu`).

### Environment

- GCP `g4-standard-48`: 4× RTX PRO 6000 Blackwell Server Edition 96 GB (CC 12.0/SM120),
  driver 610.43.02 (CUDA UMD 13.3), x86_64. Differences from the RunPod pod: single
  AMD EPYC 9B45 socket, **one NUMA node** (RunPod: 2× Xeon, GPUs split across NUMA 0/3),
  708 GiB RAM, 968 GB boot PD (no `/workspace`; no MooseFS). `python3-dev`, `python3-venv`,
  git, git-lfs preinstalled. **Shared VM**: `/home/soroosh` carries an unrelated 405 GB
  TRT-LLM project — check `nvidia-smi` for foreign GPU processes before timed runs.
- Installed by bring-up: torch 2.13.0+cu130, triton 3.7.1, transformers 5.15.0.dev0
  (editable @ `150eb7c9ed40`), model source @ `60d8d70770c6`, kernels 0.15.2, datasets
  5.0.0, huggingface_hub 1.24.0. Capture: `artifacts/env/*-20260720T064803Z.*`.

### Commands run (essentials)

```bash
# step 2 — setup_env.sh executed step-by-step (same commands; permission-classifier
# on this box blocks whole-script background runs), ending with:
.venv/bin/python tools/hardware_smoke.py                     # all required+optional PASS
.venv/bin/python tools/runpod_landing_test.py --expect configs/expectations_runpod.json --run-suite
# -> ALL 15 CHECKS PASSED; suite 95 passed (7.35 s). expectations_runpod.json passes
#    verbatim on GCP (GPU name 'NVIDIA RTX PRO 6000 Blackwell Server Edition' contains
#    'RTX PRO 6000'; CC 12.0) — no expectations_gcp.json needed.
.venv/bin/python tools/p2p_stress_check.py                   # step 3, BEFORE any multi-GPU work
# -> 0 corrupt transfers, all 12 ordered pairs, both phases -> D-014 (workaround opt-in)
HF_HUB_ENABLE_HF_TRANSFER=0 RUNPOD_ALLOW_WEIGHTS=1 \
  MODEL_DIR=/home/sukrit/models/DeepSeek-V4-Flash bash scripts/runpod/download_model.sh
# -> 149 GB / 46 shards @ 60d8d70 in ~6 min (xet backend); 289 GB disk headroom left
.venv/bin/python -m pytest tests -q                          # 98 passed (95 + 3 new)
```

### Findings / changes

1. **Native P2P is healthy on this host** → D-014: `ensure_host_staged_p2p()` is now
   opt-in via `V4_KV_FORCE_HOST_STAGED_P2P=1` (gate inside the function, env check
   before any CUDA call; `tests/test_p2p_workaround.py`). `p2p_stress_check.py
   --workaround` arms it itself. GCP numbers will reflect direct PCIe P2P — the
   RunPod "host-staged D2D" caveat is dead here.
2. Model paths moved off `/workspace` (does not exist on GCP): weights at
   `/home/sukrit/models/DeepSeek-V4-Flash`; defaults updated in
   `configs/bench_runpod_4gpu.json`, `tools/run_calibration_full.py`,
   `scripts/runpod/download_model.sh`.
3. `download_model.sh` no longer defaults `HF_HUB_ENABLE_HF_TRANSFER=1` (hub 1.x
   dropped hf_transfer — B1 finding; xet is the default backend and delivered the
   149 GB in ~6 min on GCP's NIC).
4. **B3 gates driver committed** as `tools/b3_identity_gates.py` (reconstructed from
   WORKLOG B3 / `artifacts/phase_b_runpod/b3_gates_run2.log`; the original was pod
   scratchpad) — step 5b and any future host re-validation now have a repo tool.

### Next step

Step 5 re-validation (one model load each): B1 generation sanity, then
`tools/b3_identity_gates.py`. Then B4 run 1.

## 2026-07-17 (RunPod Phase B — B2: baseline benchmark bring-up)

### Goal

Plan step B2: baseline memory/timing on the full model (`configs/bench_runpod_4gpu.json`,
`--variants baseline`).

### Findings

1. **One-shot 65536-token prefill OOMs a 96 GB card** under eager attention (CSA layers
   score against ~16k compressed entries → ~20 GiB transients on GPU 0 on top of 34 GiB
   weights). Fix: `prefill_chunk` in the bench engine (chunked prefill through the same
   cache; semantics test-pinned since Task 01; TTFT = full chunked wall time) +
   per-cell OOM resilience so a failed cell no longer discards the matrix. runpod config
   uses `prefill_chunk: 2048`. New test: chunked == one-shot byte accounting. Suite 91.
2. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is unusable on this stack**
   (torch 2.13.0+cu130, SM120, 4-GPU device_map): it causes *spurious* device-side
   asserts (`vectorized_gather_kernel: index out of bounds`) on provably in-bounds
   embedding gathers — first forward, 1024 random-id prompt. Eliminated by A/B: same
   command passes without the flag, crashes with it; token-id ladder (0..129279),
   length ladder, and runtime table audits all clean. The flag had been added
   speculatively during OOM debugging — one-variable-at-a-time relaunches would have
   found this in one cycle instead of four model loads. Do not set it.
3. Cell sanity numbers (1 trial, no warmup, cold autotune — NOT the report numbers):
   prompt 1024 baseline TTFT 8.1 s, decode 0.62 tok/s (ITL p50 334 ms; includes
   first-call Triton tuning and host-staged P2P handoffs). Real medians come from the
   full warmed run.

### Results (medians of 5, warmup 2, prefill_chunk 2048, P2P workaround active)

| prompt | TTFT | prefill tok/s | decode tok/s | ITL p50 | cache bytes | peak alloc (hottest GPU) |
|---|---|---|---|---|---|---|
| 1024 | 546 ms | 1874 | 3.3 | 296 ms | 13.4 MiB | 41.1 GiB |
| 8192 | 5.67 s | 1444 | 3.2 | 299 ms | 60.5 MiB | 46.6 GiB |
| 65536 | 88.4 s | 742 | 2.6 | 381 ms | 437 MiB | 67.8 GiB |

Micro-overheads at 65k-context shapes: fp8 window-step encode 146 µs; fp8 full window /
compressed decode 35 / 57 µs; fp4 indexer entry encode 370 µs, full decode 124 µs.
Caveats: decode dominated by host-staged D2D (D-011) + Triton fallback (DeepGEMM off in
multi-device); allocator OOM-retry warnings at 65k (peak near ceiling) — completed, but
65k is close to this pipeline's memory limit. Same-node comparisons only.
Output: `results/benchmark_cache.json`.

### Next step

B3 gates (identity + D-009 CUDA re-check) — see next entry.

## 2026-07-17 (RunPod Phase B — B3 gates: PASSED after ldexp fix)

**Final result (post-fix rerun): ALL GATES PASSED.** On both a natural and a
random-id prompt (prefill ~512, 8 teacher-forced steps): gate A —
QDQCache(baseline_bf16) bitwise-identical logits AND indexer picks vs stock cache;
gate B (D-009 CUDA/BF16 re-check on the real model) — QuantizedStorageCache ==
QDQCache bitwise under `reference_official_qdq`. Stage-B/Stage-C machinery is fully
validated on the real 4-GPU checkpoint.

**Root cause of the earlier IMA: upstream PyTorch bug.** `torch.ldexp`
(2.13.0+cu130) lacks a CUDA device guard — on tensors whose device ≠ current device
it returns NaN/garbage or IMAs (standalone 10-line repro, no model). Hit via
`ceil_pow2` the moment QDQ ran on layers mapped to cuda:1-3; invisible on single-GPU
Phase A and CPU. Fixed in `qdq.py` by IEEE-754 bit-construction (bit-exact vs ldexp
on CPU; multi-device stress clean; suite 91 passed). TODO operator: report upstream.

Informative (not a gate): official-QDQ vs baseline max|Δlogit| ≈ 7.5 (natural) /
4.8 (random) — dominated by indexer top-k flips at near-ties, the exact failure mode
D-004 predicts; per D-004 quality is judged by top-k overlap + KL/NLL on held-out
data (B5), not max-logit.

## 2026-07-17 (RunPod Phase B — B3 gates, earlier notes)

Driver: scratchpad `b3_identity_gates.py` (gate A: QDQCache(baseline_bf16) ==
baseline bitwise; gate B: storage == qdq under reference_official_qdq; on a natural and
a random-id prompt, prefill ~512 + 8 teacher-forced steps).

**Gate A (natural prompt): PASS — logits AND indexer picks bitwise identical** on the
real 4-GPU model. First hard evidence the injection machinery is numerically invisible
on the real checkpoint.

**Crash before gate B**: `cudaErrorIllegalAddress` (async, surfaced at cache
`lazy_initialization .to(device)`) during the FIRST `reference_official_qdq` QDQ run —
the first time the QDQ cache subclasses + indexer-query wrapper run on a MULTI-DEVICE
model (Phase A validated single-GPU only). No module-level device-pinned constants in
`qdq.py`/`qdq_cache.py` (checked). Next action: rerun failing leg alone under
`CUDA_LAUNCH_BLOCKING=1` to get the true faulting op; audit QDQ cache-layer code for
tensors created on a fixed device instead of the incoming tensor's device.

Blocking rerun: IMA surfaces at `qdq.py ceil_pow2 (torch.ldexp)` in the FIRST fp8 cache
write — but ldexp/QDQ ran clean on CUDA in Phase A (single GPU), and Triton launches
bypass LAUNCH_BLOCKING, so working theory: a preceding Triton finegrained-fp8 kernel
reads/writes out of bounds, and visibility depends on allocator layout (this may ALSO
explain the earlier expandable_segments "spurious" gather asserts — garbage ids from an
OOB write, same underlying bug, different victim). Gate A passing bitwise twice says
baseline paths are clean; the trigger is specific to the QDQ leg's allocation pattern
or its aten/Triton interleaving. Bisect plan (one process per leg, ~15 min each):
(1) policy `main_fp8_nonrope_rope_bf16` (fp8 main KV only, no indexer wrapper);
(2) policy `indexer_reference_qdq` (indexer only). Then compute-sanitizer on the
guilty component if needed.

## 2026-07-17 (RunPod Phase B — B1: weights, generation gate, P2P fault investigation)

### Goal

Plan steps B1 (`docs/RUNPOD_PHASE_B_PLAN.md`): download weights, run the untouched-model
generation sanity gate (the definitive native-FP8/FP4-on-SM120 check).

### Commands run (essentials)

```bash
.venv/bin/pip install -U "huggingface_hub"          # hf CLI (pre-flight deviation 2)
HF_HUB_ENABLE_HF_TRANSFER=0 RUNPOD_ALLOW_WEIGHTS=1 bash scripts/runpod/download_model.sh
# hub 1.x dropped hf_transfer; xet backend used instead. 149 GB / 46 shards @ 60d8d70.
.venv/bin/pip install "kernels==0.15.2"             # required by native FP8 path (new dep)
# generation + a chain of diagnostic forwards (scratchpad scripts, logs preserved)
.venv/bin/python tools/p2p_stress_check.py          # NEW pod-health tool -> CORRUPT
.venv/bin/python tools/p2p_stress_check.py --workaround   # -> 0 corrupt
```

### Findings

1. **Weight loading off the MooseFS volume needs cache pre-warming.** safetensors mmap
   page-faults over FUSE: load ETA was 4.5 h; after `cat`-warming all shards into page
   cache (45 s at ~3.3 GB/s, 8 parallel readers; volume does 520 MB/s cold / 7 GB/s
   warm), the 4-GPU load takes ~12 min. Use parallel pre-read before any full-model run.
2. **`kernels==0.15.2` is a required dependency** for native-FP8 inference
   (transformers' finegrained-fp8 integration loads `kernels-community/finegrained-fp8`
   at first forward). Not caught by Phase A (tiny model has no FP8 weights). Added to
   `scripts/runpod/setup_env.sh`.
3. **First generation produced garbage** (32× `<|begin_of_sentence|>`), then a chain of
   diagnostics with mutually contradictory symptoms across identical runs: NaN logits;
   1e36-magnitude (finite) q/kv at layer 0 overflowing bf16 scores into NaN; a fully
   clean run. Eliminated by direct test: FP8/FP4 Triton kernels (both `matmul_2d` and
   `matmul_grouped` bit-consistent with a dequant reference at all V4 shapes, ue8m0 and
   fp32 scales, all 12 autotune configs, incl. with a NaN-poisoned allocator pool);
   checkpoint bytes (sane scales/weights); loader (GPU buffers byte-match shards);
   masks/sinks (sign-safe by construction, no missing keys).
4. **Root cause: the pod's PCIe P2P silently corrupts GPU-to-GPU copies** — 15-30/30
   failures at 64-256 MiB idle, 20/20 on all 12 ordered pairs when copies overlap
   compute (the `device_map="auto"` regime). D2H/H2D clean, so every single-GPU test
   passed. Host ACS/IOMMU class fault; invisible to nvidia-smi. See D-011.
5. **Mitigation validated:** disabling CUDA peer access (host-staged D2D) →
   0/240 corrupt in the worst-case stress. Implemented as
   `src/v4_kv_quant/p2p_workaround.ensure_host_staged_p2p()` + health check
   `tools/p2p_stress_check.py`; generation gate rerun with workaround (result below).
6. Upstream cross-check: transformers pin is 10 commits behind HEAD with zero changes
   to deepseek_v4 / finegrained_fp8 / masking — nothing to borrow; the V4 generation
   integration test is manual (`RUN_SLOW`), so this path has little upstream mileage.

### B1 gate result

**GO.** With the P2P workaround, untouched-model greedy generation (native FP8/FP4,
`device_map="auto"`, eager, 32 tokens) produces a coherent continuation:

> "Pipeline parallelism in ai is  a technique that splits a model across multiple
> devices, with each device responsible for a subset of layers. This is different from
> data parallelism, where each device has a"

Semantically matches the upstream integration-test snapshot (which uses *dequantized
BF16*; exact-string drift expected and upstream marks its own test `@is_flaky`).
Load 868 s (warm cache), generate 26 s / 32 tokens (includes first-call kernel
autotune; not a performance number). Per-GPU allocated: 34.3/40.0/40.0/31.0 GiB —
native FP8/FP4 checkpoint fits with ~55 GiB headroom per card. Weights remain in
native dtypes on GPU (e4m3 + e8m0 scales + packed-FP4 int8).

### Next step

B1 verdict + tick, then B2 baseline benchmark (with `ensure_host_staged_p2p()` wired
into the benchmark path), and report the faulty node to RunPod (D-011 follow-up).

## 2026-07-17 (RunPod Phase B — B0 environment rebuild, 4-GPU pod)

### Goal

Execute plan step B0 (`docs/RUNPOD_PHASE_B_PLAN.md`): full environment rebuild on the
Phase-B pod — this pod is NOT the frozen Phase-A image (`.venv/`, `vendor/`, weights all
absent; only the repo was on this volume).

### Environment

- Pod: 4× RTX PRO 6000 Blackwell Server Edition 96 GB (CC 12.0/SM120), x86_64,
  2× Xeon 6952P (384 threads), 1.5 TiB RAM, driver 580.126.20, no NVLink (GPU0 on NUMA0,
  GPUs 1–3 on NUMA3), /workspace = MooseFS network volume (~94 TB free).
- Installed by bootstrap: torch 2.13.0+cu130, triton 3.7.1, transformers 5.15.0.dev0
  (editable at pin `150eb7c9ed40`), model source at `60d8d70770c6` (13 MiB LFS pointers).
  Capture: `artifacts/env/*-20260717T163605Z.*`.

### Commands run

```bash
bash scripts/runpod/setup_env.sh   # full run: venv+torch+vendor pins+capture+smoke+landing+suite
```

### Tests and results

- Hardware smoke: all required + all optional PASS (BF16/FP8/FP4 dtypes CPU+CUDA,
  torch.compile, hand-written Triton kernel, `torch._scaled_mm` FP8 GEMM).
- Landing test vs `configs/expectations_runpod.json`: **all 15 checks PASS**, suite
  **90 passed** (99.6 s) → `ALL CHECKS PASSED`, exit 0. Report: `results/landing_test.json`.
- No incompatibilities; no code changes needed.

### Next step

B1: install `hf` CLI (absent on this pod — pre-flight deviation 2), guarded weight
download (~160 GB, pinned revision), then baseline generation sanity = the GO/NO-GO
gate for native FP8/FP4 on SM120.

## 2026-07-17 (RunPod Phase A — 1-GPU landing pod, environment + portability gate)

### Goal

Execute `RUNPOD_START_HERE.md` Phase A on the landing pod: rebuild the environment for
x86-64/SM120, pass hardware smoke + landing test + full suite, confirm the benchmark
measurement path on CUDA, record incompatibilities.

### Environment

- Pod: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB (compute capability 12.0 /
  SM120), x86_64, 256 vCPU / 1.5 TB RAM, driver 580.126.09, network volume on /workspace.
- Image ships CUDA toolkit 12.8 (nvcc) — irrelevant at runtime: the cu130 torch wheel
  bundles the CUDA 13.0 runtime and triton bundles its own ptxas. Default
  `TORCH_INDEX_URL` (cu130) worked; no override needed.
- Python 3.12.3 **with python3.12-dev preinstalled** — the GX10 blocker (Triton-backed
  `torch.bmm` needs `Python.h`) does not exist here; CUDA model execution works.
- Installed: torch 2.13.0+cu130, triton 3.7.1, transformers 5.15.0.dev0 (editable at pin
  `150eb7c9ed40`), model source at `60d8d70770c6` (13 MiB LFS pointers, no weights).
  Capture: `artifacts/env/*-20260717T075537Z.*`.
- Repo state: clean checkout 4 doc-only commits after `dgx-phase-complete-v1`.

### Commands run

```bash
bash scripts/runpod/setup_env.sh   # venv+torch+pins+smoke+landing(+suite); first run: suite 2 failed, 88 passed
.venv/bin/python -m pytest tests/test_benchmark.py -q   # after test fix: 8 passed
.venv/bin/python tools/runpod_landing_test.py --expect configs/expectations_runpod.json --run-suite
                                   # ALL CHECKS PASSED (exit 0), suite 90 passed
.venv/bin/python tools/benchmark_cache.py --config configs/bench_tiny_local.json --device cuda   # exit 0
```

### Files changed

- `tests/test_benchmark.py` — the only incompatibility found (see Findings 1)
- `docs/RUNPOD_HANDOFF_CHECKLIST.md` (landing-test boxes ticked), `PROJECT_STATUS.md`,
  this file

### Tests and results

- Hardware smoke: **all required + all optional capabilities PASS** — BF16/FP8
  e4m3/e8m0/FP4-x2 dtypes and round-trips on SM120, `torch._scaled_mm` FP8 GEMM,
  hand-written Triton kernel, `torch.compile` (inductor/triton) on CUDA.
- Landing test: all 15 checks PASS (platform, GPU identity, CC ≥ 12.0, dtypes, vendor
  pins, no weights materialized, python dev headers, tiny forward CPU+CUDA, Stage-C
  bitwise gate). Report: `results/landing_test.json`.
- Suite: first run **2 failed, 88 passed** (host-identity tests, below); after fix
  **90 passed** under `--run-suite` → `ALL CHECKS PASSED`, exit 0.
- Tiny-model CUDA benchmark (batch 2, prompts 64/256, 32 decode tokens, 5 trials,
  medians; fp32, `reference_official_qdq` policy) — machinery validation only,
  NON-TRANSFERABLE numbers: baseline/qdq/storage all ran end to end on CUDA; storage
  cache bytes 9.8 vs 22.0 KiB (prompt 64) and 18.6 vs 46.0 KiB (prompt 256) with QDQ
  == baseline bytes as designed; QDQ/storage decode ~10–20% slower than baseline
  (expected pre-fusion). Micro-overheads recorded (fp8 encode window step ~74 µs,
  fp4 indexer entry encode ~209 µs). Output: `results/benchmark_cache.json`.

### Findings

1. **Only incompatibility: the two landing-check identity tests hardcoded the GX10 as
   "this machine"** (`test_landing_checks_pass_with_local_expectations` asserted the
   GX10 expectations pass; `..._fail_cleanly_with_runpod_expectations` asserted the
   RunPod expectations fail). On the pod both inverted. Fixed by selecting the
   expectations file from `platform.machine()` — "this host's expectations PASS, the
   other host's FAIL cleanly" — so the suite is green on both hosts with unchanged
   intent. No product-code change was needed anywhere.
2. Everything Phase A is allowed to prove about SM120 is proven: FP8/FP4 dtypes and
   casts, FP8 GEMM, Triton compile path, full tiny V4 architecture execution on GPU
   (eager attention, compressors, indexer, all three cache variants, Triton-backed
   `torch.bmm`), Stage-C bitwise gate on this dtype stack. Real-checkpoint quantized
   kernels remain unprovable on 96 GB (Phase B step 1 is the definitive check).
3. cu130 wheel on a CUDA-12.8 image is fine (driver 580.x provides the 13.0 runtime
   surface); worth knowing when picking Phase B images.
4. Compute capability reports **12.0** here vs 12.1 on the GX10 GB10 — the
   `min_compute_capability: [12, 0]` expectation is correct as written.

### Blockers / open questions

- None for Phase B. Remaining Phase A checklist items are operator actions:
  freeze the pod image, stop the pod.

### Next step

Phase B (4× GPU pod): reuse frozen image; budget the first hour for step 1 baseline
generation sanity — native FP8/FP4 inference on SM120 — before committing to the full
experiment plan (fallback: BF16 dequant on 8×96 GB).

## 2026-07-17 (Local completion gate — DGX/GX10 phase closed)

### Goal

Walk the completion gate in docs/DGX_PHASE_PLAN.md and the handoff preflight, then tag.

### Commands run

```bash
.venv/bin/pip freeze > docs/PIP_FREEZE_GX10.txt
cp results/calibration_smoke/*.json artifacts/calibration_smoke/
.venv/bin/python tools/generate_results_manifest.py        # 23 files -> docs/results_manifest.json
.venv/bin/python -m pytest tests/ -q                       # 90 passed
# fresh-worktree validation (catches dependence on uncommitted files):
git worktree add /tmp/dgx-gate-check HEAD
PYTHONPATH=/tmp/dgx-gate-check/src .venv/bin/python -m pytest /tmp/dgx-gate-check/tests -q
git worktree remove /tmp/dgx-gate-check
git tag -a dgx-phase-complete-v1
```

### Files changed

- `docs/PIP_FREEZE_GX10.txt` (frozen GX10 versions; reference only)
- `docs/results_manifest.json` + `tools/generate_results_manifest.py`
- `artifacts/calibration_smoke/` (committed calibration fixtures: token ids, precision
  map, sensitivity records, held-out metrics, stats)
- `docs/DGX_PHASE_PLAN.md` (all boxes checked; gradient-weighted ranking left open as
  explicitly optional/deferred), `docs/RUNPOD_HANDOFF_CHECKLIST.md` (preflight complete),
  `PROJECT_STATUS.md`

### Tests and results

Full suite 90 passed on the working tree AND from a clean git worktree of the final
commit. No weights (vendor model tree 14 MB pointers). Working tree clean at tag time.

### Findings

Gate closed with one explicitly-deferred optional item (gradient-weighted ranking) and one
documented environment limitation (GX10 CUDA blocked pending python3.12-dev — D-010).

### Next step

RunPod one-GPU landing pod per docs/RUNPOD_HANDOFF_CHECKLIST.md.

## 2026-07-17 (Task 05 — benchmark harness + RunPod tooling, Phase 6)

### Goal

Config-driven benchmark CLIs (baseline/QDQ/storage), source-only landing test, guarded
RunPod launch scripts — one command structure for tiny-local and full-model-RunPod.

### Commands run

```bash
.venv/bin/python -m pytest tests/test_benchmark.py -q                      # 8 passed
.venv/bin/python -m pytest tests/ -q                                       # 90 passed
.venv/bin/python tools/benchmark_cache.py --config configs/bench_tiny_local.json
.venv/bin/python tools/runpod_landing_test.py --expect configs/expectations_gx10.json    # exit 0
.venv/bin/python tools/runpod_landing_test.py --expect configs/expectations_runpod.json  # exit 1 here (correct)
bash scripts/runpod/setup_env.sh                    # REFUSING (aarch64 guard) - correct
RUNPOD_ALLOW_WEIGHTS=1 bash scripts/runpod/download_model.sh   # REFUSING - correct
```

### Files changed

- `prompts/05_BENCHMARK_RUNPOD_TOOLING.md`
- `src/v4_kv_quant/bench.py` (BenchSettings from JSON; identical token streams across
  variants; warmup+trials; TTFT/prefill/decode/ITL; per-GPU peaks; cache bytes; QDQ
  overhead micro-bench), `src/v4_kv_quant/landing.py` (expectation-driven checks)
- `tools/benchmark_cache.py`, `tools/runpod_landing_test.py`
- `configs/{bench_tiny_local,bench_runpod_4gpu,expectations_gx10,expectations_runpod,source_pins}.json`
- `scripts/runpod/{setup_env,download_model,launch_4gpu_bench}.sh` (x86_64-guarded;
  weights additionally behind RUNPOD_ALLOW_WEIGHTS=1 + free-disk check + pinned revision)
- `tests/test_benchmark.py`; `docs/REPRODUCIBILITY.md` limitation 1 upgraded

### Tests and results

90/90 pass. Local tiny benchmark (CPU, fp32, non-transferable): storage cache 9.8/18.6 KiB
vs baseline 22/46 KiB at prompt 64/256; QDQ==baseline bytes; quantized variants slower as
expected for pure-PyTorch (no claims). Landing test: GX10 expectations all PASS
(2 truthful WARNs), RunPod expectations fail on platform/GPU/dev-headers here — proving
the expectation mechanism.

### Findings

1. **GX10 CUDA model execution is blocked by the missing python3.12-dev**: torch 2.13
   routes CUDA `torch.bmm` through a Triton-backed `torch._native` kernel, and V4's
   grouped output projection is bmm-based — first CUDA forward of the tiny model
   surfaced it. CPU unaffected. Landing test now carries an expectation-driven
   `cuda_model_forward` check (WARN on GX10, FAIL on RunPod); local bench config pins CPU.
2. CUDA peak-stats APIs error if the CUDA context is uninitialized — benchmark guards all
   `torch.cuda` calls on the *benchmark device*, not on `is_available()`.

### Blockers / open questions

- `python3.12-dev` install (system package) needed before ANY GPU-side validation on the
  GX10; decision deferred to the owner (D-005/D-010). Not required for handoff — RunPod
  images ship dev headers and the landing test enforces it there.

### Next step

Local completion gate: freeze, manifest, tag `dgx-phase-complete-v1`, handoff preflight.

## 2026-07-17 (Task 04 — Stage-C actual-storage prototype, Phase 5)

### Goal

Real low-precision cache storage (FP8 codes + e8m0 scales; packed FP4 nibbles) with
pure-PyTorch dequantize-on-read, bitwise-faithful to the Stage-B QDQ simulation, plus
honest memory accounting demonstrating actual byte reduction.

### Commands run

```bash
.venv/bin/python -m pytest tests/test_actual_storage.py -q   # 15 passed (first run)
.venv/bin/python -m pytest tests/ -q                         # 82 passed
.venv/bin/python tools/measure_cache_memory.py               # results/cache_memory.json
```

### Files changed

- `prompts/04_ACTUAL_STORAGE_PROTOTYPE.md` (spec + gates)
- `src/v4_kv_quant/storage.py` (fp8_store/fp4_store/load with bitwise QDQ parity;
  e8m0 1-byte scales for power-of-2, fp32 fallback; sign+3-bit e2m1 codes, 2 per uint8)
- `src/v4_kv_quant/storage_cache.py` (QuantStore append/trim/decode; storage layers for
  sliding/HCA/CSA; QuantizedStorageCache; keys/values stay empty placeholders)
- `src/v4_kv_quant/memory.py` (logical + allocator storage bytes; K=V alias once;
  stock sliding V-duplication flagged; per-state itemization)
- `src/v4_kv_quant/harness.py` (`storage=True` runs the Stage-C cache for a policy)
- `tools/measure_cache_memory.py`, `tests/test_actual_storage.py`

### Tests and results

82/82 pass. Memory comparison (tiny fp32 model, reference_official_qdq, 112 tokens):

| cache | logical bytes | vs baseline |
|---|---|---|
| baseline_bf16_stock | 24,576 | 1.000x |
| stage_b_qdq_sim | 24,576 | 1.000x (simulation saves nothing — proven, not claimed) |
| stage_c_storage | 10,758 | **0.438x** (per-layer 0.223 / 0.485 / 0.445) |

Itemization confirms: fp8 codes 1 B, e8m0 scales 1 B/group, raw fp32 rope slices, packed
uint8 indexer nibbles (2, 28, 8) for 16 logical channels, fp32 buffers/overlap intact.
Sliding layer gains extra from removing the stock duplicated V copy.

### Findings

1. **Stage-C == Stage-B bitwise at model level** (logits and indexer picks) for both
   fp8-main and full official policies — every Stage-B quality result transfers to real
   storage unchanged. This is the load-bearing Stage-C correctness gate (D-009).
2. e8m0/fp8 tensor dtypes support cat/slice/contiguous on CPU in torch 2.13 — no uint8
   view workarounds needed for storage bookkeeping.
3. Window trim must re-contiguate: the stock dynamic layers keep views into concatenated
   history (storage_bytes > logical_bytes, visible in the report as 35,840 vs 24,576);
   Stage-C stores are trimmed contiguous, so allocator bytes track logical bytes.
4. fp32 tiny-model ratios overstate real savings (fp8 vs fp32 = 4x on quantized slices);
   BF16 checkpoint gives 2x there. Accounting validated; final ratios come from RunPod.

### Blockers / open questions

None. Decode-path latency is expectedly worse than baseline (pure-PyTorch dequant per
forward); not measured on GX10 by design — Stage-D fusion is target-hardware work.

### Next step

Task 05: benchmark harness + RunPod tooling (Phase 6), then the local completion gate.

## 2026-07-16 (Task 03 — calibration and precision-policy plumbing, Phase 4)

### Goal

Target taxonomy, per-group precision map + consumer cache, activation stats, one-target
empirical perturbation sweep, map builder, smoke calibration with held-out evaluation.

### Commands run

```bash
.venv/bin/python -m pytest tests/test_calibration.py -q   # 13 passed (first run)
.venv/bin/python -m pytest tests/ -q                      # 67 passed
.venv/bin/python tools/run_calibration_smoke.py           # results/calibration_smoke/
```

### Files changed

- `prompts/03_CALIBRATION_PLUMBING.md` (spec + gates)
- `src/v4_kv_quant/targets.py` (QuantTarget, enumerate_targets; indexer = state-level target)
- `src/v4_kv_quant/precision_map.py` (MapEntry/PrecisionMap v1, validation, JSON)
- `src/v4_kv_quant/mapped_cache.py` (MappedQDQCache + per-group write-boundary QDQ +
  indexer query context from map)
- `src/v4_kv_quant/stats.py` (pass-through StatsCollectorCache: amax/mean|x|/RMS,
  per-group amax, FP8/rotated-FP4 QDQ-error RMS)
- `src/v4_kv_quant/sensitivity.py` (measure_target, run_sensitivity_sweep,
  build_map_from_sweep with fp8/fp4 fractions + indexer overlap threshold)
- `src/v4_kv_quant/harness.py` (accepts `precision_map=` alongside `policy=`)
- `tools/run_calibration_smoke.py`, `tests/test_calibration.py`

### Tests and results

67/67 pass. Smoke run (tiny random model, 16 targets, group_size_main=8, S=48/prefill 24):

- stats: indexer rotated-FP4 QDQ error RMS 1.1e-1 vs main-KV FP8 2.5–2.8e-2 (~4x) — expected;
- sensitivity ranking: indexer target most sensitive (score 1.6e-3, ~4x above the top main
  target); layer-0 window groups next (earliest layer, most downstream amplification);
  compressed-KV groups least sensitive on this random model;
- built map: 11x fp8_e4m3 + 1x indexer fp4_hadamard, 4 most-sensitive groups left BF16;
- held-out eval: top-1 agreement 0.9375, KL 6.3e-5, dNLL -4.4e-4, no NaN/Inf.

All numbers are random-weight machinery validation — not transferable to the checkpoint.

### Findings

1. Single-entry precision map == perturbation experiment; full map == deployable policy —
   one consumer (`MappedQDQCache`) serves calibration and deployment, cross-validated
   bitwise against the Task-02 policy cache on full coverage.
2. Indexer calibrated as ONE state-level target: Hadamard rotation mixes all channels, so
   per-group granularity in the original basis has no deployable meaning (D-008).
3. Gradient-weighted ranking deferred (optional per CLAUDE.md); empirical perturbation is
   the primary truth and the harness makes each target measurement ~1 s on the tiny model.

### Blockers / open questions

None. Real-corpus calibration data (C4 port) and real-weight sensitivity are RunPod work.

### Next step

Task 04: Stage-C actual-storage prototype (Phase 5) — see PROJECT_STATUS.md.

## 2026-07-16 (Task 02 — official-policy QDQ simulation, Stage B)

### Goal

Reproduce the official QAT-aligned QDQ numerics at the Transformers cache write boundary,
selected purely by a serializable policy. Simulation only — no memory savings.

### Commands run

```bash
.venv/bin/python -m pytest tests/test_qdq_simulation.py -q   # 27 passed
.venv/bin/python -m pytest tests/ -q                         # 54 passed
.venv/bin/python tools/run_qdq_simulation.py                 # results/qdq_simulation.json
```

### Files changed

- `prompts/02_OFFICIAL_POLICY_QDQ_SIMULATION.md` (task spec + acceptance gates)
- `src/v4_kv_quant/qdq.py` (FP8 e4m3 QDQ g64 + exact ue8m0 round-up scales via frexp;
  software FP4 e2m1 RNE g32; orthonormal FWHT; effective_group_size with no silent fallback)
- `src/v4_kv_quant/policy.py` (StatePolicy/KVQuantPolicy v1, JSON round trip, 5 presets)
- `src/v4_kv_quant/qdq_cache.py` (QDQSlidingWindowLayer / QDQHCACacheLayer / QDQCSACacheLayer,
  QDQCache container, indexer-query scorer wrapper context manager)
- `src/v4_kv_quant/{metrics,harness}.py` (logit/KL/NLL/top-1 + indexer top-k overlap;
  teacher-forced runner)
- `tools/run_qdq_simulation.py`, `tests/test_qdq_simulation.py`

### Tests and results

54/54 pass. Tool output (tiny RANDOM-weight model, fp32 CPU, S=48, prefill 24 — machinery
validation and relative ordering only):

| policy | max abs dlogit | KL mean | top-1 agree | indexer overlap |
|---|---|---|---|---|
| baseline_bf16 | 0.0 (bit-exact) | 0.0 | 1.0000 | 1.0000 |
| main_fp8_nonrope_rope_bf16 | 3.8e-2 | 7.9e-6 | 0.9688 | 0.9944 |
| reference_official_qdq | 1.7e-1 | 5.3e-5 | 0.9479 | 0.9444 |
| main_fp4_nonrope_rope_bf16 (exp) | 1.8e-1 | 1.4e-4 | 0.8958 | 0.9389 |
| indexer_reference_qdq | 1.2e-1 | 4.7e-5 | 0.9583 | 0.9333 |

Ordering is sensible: FP8 mildest; FP4-on-main worst; indexer FP4 shows up mainly as top-k
overlap loss (0.93) — consistent with the Task-01 tie-instability finding.

### Findings

1. torch 2.13 cannot cast to `float4_e2m1fn_x2` (storage-only; "copy_kernel not implemented")
   → software e2m1 grid with RNE ties (tie table 0.25→0, 0.75→1, 1.25→1, 1.75→2, 2.5→2,
   3.5→4, 5→4), verified idempotent and grid-exact.
2. `CacheLayerMixin.__init_subclass__` re-registers subclasses by inherited `_layer_type` —
   QDQ subclasses MUST set `_layer_type = None` or they silently hijack every
   `DynamicCache(config=…)` in the process (test-pinned).
3. Indexer entries are stored in the Hadamard-ROTATED basis and queries rotated symmetrically
   via a scorer wrapper — matches official model.py:414-420; FWHT dot-product preservation
   verified to 1e-4.
4. QDQ at the write boundary is chunking-invariant: chunked/one-shot/token-by-token agree to
   Task-01 tolerances under FP8 policy (per-token row groups + per-entry emission).

### Blockers / open questions

None for local work. Real-checkpoint validation of all quality numbers is a RunPod gate.

### Next step

Task 03: calibration plumbing (Phase 4) — see PROJECT_STATUS.md.

## 2026-07-16

### Goal

Execute Task 01 (`prompts/01_ARCHITECTURE_DISCOVERY.md`): normalize the workspace, pin
sources, map the DeepSeek-V4 cache architecture from executable code, and protect the
baseline semantics with deterministic tiny-model tests. No weights, no quantization code.

### Commands run

```bash
# workspace (user-approved layout: starter promoted to repo root)
mv deepseek-v4-kv-dgx-starter/* . && rmdir deepseek-v4-kv-dgx-starter
mv kv-cache-quantization-kimi-k27-main reference/v2_mla_poc && chmod -R a-w reference/v2_mla_poc
bash scripts/capture_environment.sh                      # pre-venv snapshot

# sources (SHAs in docs/REPRODUCIBILITY.md; weights = LFS pointers only, verified)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash vendor/DeepSeek-V4-Flash
git clone --depth 50 https://github.com/huggingface/transformers.git vendor/transformers

# environment (project venv only; system untouched)
python3 -m venv .venv && .venv/bin/pip install -U pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130   # 2.13.0+cu130, CUDA OK on GB10
.venv/bin/pip install numpy safetensors pytest accelerate sentencepiece
.venv/bin/pip install -e vendor/transformers && .venv/bin/pip install -e .
PATH="$PWD/.venv/bin:$PATH" bash scripts/capture_environment.sh   # post-venv snapshot

# deliverables
.venv/bin/python tools/inspect_v4_cache.py               # all runtime assertions PASS
.venv/bin/python tools/hardware_smoke.py                 # required PASS; triton/compile UNSUPPORTED
.venv/bin/python -m pytest tests/test_v4_cache_semantics.py -q    # 27 passed in ~5s
```

### Files changed

- `docs/V4_CACHE_ARCHITECTURE.md`, `docs/REFERENCE_PORT_MAP.md`,
  `docs/QUANTIZATION_INJECTION_PLAN.md`, `docs/REPRODUCIBILITY.md` (new)
- `src/v4_kv_quant/{__init__,tiny_model}.py`, `pyproject.toml` (new package)
- `tools/inspect_v4_cache.py`, `tools/hardware_smoke.py` (new)
- `tests/test_v4_cache_semantics.py` (new, 27 tests)
- `PROJECT_STATUS.md`, `docs/DECISIONS.md`, this file
- `.gitignore` (+`vendor/`), `README.md` (starter version at root)

### Tests and results

- `pytest tests/test_v4_cache_semantics.py`: **27 passed** (fp32 CPU, deterministic seeds).
- `tools/inspect_v4_cache.py`: 29/29 runtime assertions PASS (counts, widths, K=V sharing).
- `tools/hardware_smoke.py`: BF16 / FP8 e4m3 / e8m0 / FP4-x2 dtypes PASS on CPU+GB10;
  `torch._scaled_mm` PASS; `torch.compile` + Triton UNSUPPORTED (missing python3.12-dev —
  root-caused via manual gcc repro; system package deliberately not installed).

### Findings

1. V4 replaces MLA entirely: shared-KV MQA (K=V, 512 = 448 nope + 64 rope, inverse RoPE on
   output), per-layer sliding window 128, CSA (m=4 + top-k-512 indexer) / HCA (m'=128)
   compressed entries appended on the KV axis inside attention.
2. HF cache states per compressed layer: window `keys`(=values), `compressed_kv`,
   `buffer_kv/gate` (< rate tokens), CSA `overlap_kv/gate` (one window Ca slice),
   `entry_count`; all writes flow through `update` / `store_compression_weights` /
   `update_compressor_states` → quantization injects as cache-layer subclasses.
3. Official QDQ policy extracted from `inference/{model,kernel}.py`: FP8 e4m3 g64, ue8m0
   round-up power-of-2 scales, non-rope dims only; indexer: Hadamard + FP4 e2m1 g32 on keys
   AND queries; buffers fp32; rope dims BF16. HF impl has zero QDQ (that gap = Task 02).
4. **Measured**: batched-prefill vs token-by-token decode logits diverge (up to ~1e-1) with
   the selective indexer purely from top-k tie flips under 1e-7 float noise; agree to 2.4e-7
   with a non-selective indexer; MoE routing ruled out. Equality tests therefore use a
   dense-indexer fixture; indexer quantization metric = top-k overlap/recall.
5. `use_cache=True` vs `False` prefill logits are bit-identical; chunked prefill of any
   tested chunking matches one-shot within 2.4e-7 (dense indexer).
6. Stock `DynamicSlidingWindowLayer` (sliding-only layers) duplicates V storage; V4's own
   cache layers alias K=V — a free 2x saving on 3 layers for the Stage-C custom layer.
7. Upstream-documented constraints confirmed in source: eager-attention only (FA head-dim
   cap 256 < 512), left padding unsupported by design, `QuantizedCache` incompatible,
   compressor state non-rewindable (`_is_stateful`).

### Blockers / open questions

- No blockers for Task 02.
- Open (deferred to RunPod): HF<->official numerical parity on real weights; e8m0-vs-fp32
  scale quality; exact-Hadamard vs `fast_hadamard_transform`; YaRN at real context lengths.

### Next step

Task 02: official-policy QDQ simulation as policy-configurable cache-layer subclasses
(scope in `docs/QUANTIZATION_INJECTION_PLAN.md`, gate list in `PROJECT_STATUS.md`).
