# Decision Log

Record architecture and experiment decisions here. Do not leave consequential decisions only in chat or terminal history.

### D-001 — Promote starter pack to repository root

**Date:** 2026-07-16
**Status:** accepted (user-approved)
**Context:** The git repo initially held the starter pack and the unzipped reference repo as
sibling directories; CLAUDE.md/START_HERE.md expect starter contents at the project root with
the reference at `reference/v2_mla_poc/`.
**Decision:** Move starter contents to the repo root; move the reference repo to
`reference/v2_mla_poc` and `chmod -R a-w`.
**Alternatives considered:** keep starter as a subdirectory and nest everything inside it.
**Evidence:** commit `9b0825b`.
**Consequences:** CLAUDE.md auto-loads at the repo root; all documented paths are canonical.
**Follow-up:** none.

### D-002 — Project venv with cu130 SBSA torch + editable pinned transformers

**Date:** 2026-07-16
**Status:** accepted
**Context:** No system torch/transformers exist; system-wide changes are prohibited during
discovery; GB10 is aarch64 with CUDA 13 / cc 12.1.
**Decision:** `python3 -m venv .venv`; `torch 2.13.0+cu130` from the cu130 wheel index
(CUDA verified working on GB10); transformers installed **editable from the pinned
`vendor/transformers` checkout** so tests exercise exactly the pinned revision; `vendor/`
gitignored with SHAs recorded in `docs/REPRODUCIBILITY.md`.
**Alternatives considered:** CPU-only torch (kept as fallback, not needed); uv (not installed);
tracking vendor sources in-repo (bloat, nested-git issues).
**Evidence:** `docs/REPRODUCIBILITY.md`; `tools/hardware_smoke.py` results.
**Consequences:** ARM64 venv is non-portable — RunPod rebuilds from the recorded pins
(CLAUDE.md constraint 5).
**Follow-up:** freeze full `pip freeze` at DGX-phase completion.

### D-003 — Quantization injects as cache-layer subclasses, not attention patches

**Date:** 2026-07-16
**Status:** accepted
**Context:** All V4 cache writes flow through three methods on per-layer cache classes
(`update`, `store_compression_weights`, `update_compressor_states`); the reference PoC's
attention-forward monkey-patch pattern targets attributes V4 doesn't have; upstream marks
generic `QuantizedCache` incompatible.
**Decision:** Implement QDQ (Stage B) and real storage (Stage C) as subclasses of
`DeepseekV4CSACache` / `DeepseekV4HCACache` / `DynamicSlidingWindowLayer`, constructed from a
serializable precision policy; never edit generated modeling files. The single exception:
indexer *query* QDQ wraps `DeepseekV4Indexer` externally.
**Alternatives considered:** forward-patching attention (fragile, rejected); editing
`modular_deepseek_v4.py` + regenerating (upstream-invasive, rejected for experiments).
**Evidence:** `docs/V4_CACHE_ARCHITECTURE.md` §2; `docs/QUANTIZATION_INJECTION_PLAN.md`.
**Consequences:** experiments select behavior purely by constructing the cache; baseline path
untouched.
**Follow-up:** Task 02 implements the policy object + subclasses.

### D-004 — Indexer quantization judged by top-k overlap, not logit closeness

**Date:** 2026-07-16
**Status:** accepted
**Context:** Measured on the tiny model: near-tied indexer scores flip the top-k set under
~1e-7 numerical noise, producing ~1e-1 logit divergence between mathematically equivalent
computation paths; with a non-selective indexer the same paths agree to 2.4e-7.
**Decision:** Path-equality tests use a dense-indexer fixture (`index_topk` ≥ entry count);
indexer quantization quality is measured by top-k overlap/recall + downstream NLL/KL, matching
CLAUDE.md calibration principles.
**Alternatives considered:** loosening logit tolerances (hides real regressions); forcing
deterministic tie-breaking upstream (diverges from shipped behavior).
**Evidence:** isolation experiment in `docs/V4_CACHE_ARCHITECTURE.md` §6.8; test-file docstring.
**Consequences:** Task 02 metrics suite must include a top-k overlap harness.
**Follow-up:** validate flip rates on real weights on RunPod (random weights overstate ties).

### D-005 — Do not install python3.12-dev during discovery; defer Triton/torch.compile

**Date:** 2026-07-16
**Status:** accepted
**Context:** Triton and inductor fail at gcc because `/usr/include/python3.12/Python.h` is
missing (system package). Installing it is a system-wide change, prohibited by CLAUDE.md
constraint 4 / task instructions; Stages A–B are pure PyTorch by design.
**Decision:** Record as documented limitation (`docs/REPRODUCIBILITY.md`,
`tools/hardware_smoke.py` report) and proceed without it.
**Alternatives considered:** `sudo apt install python3.12-dev` (deferred until a Triton
prototype actually needs it, with explicit sign-off).
**Evidence:** manual gcc reproduction; hardware_smoke UNSUPPORTED entries with root cause.
**Consequences:** no fused-kernel prototyping on GX10 until resolved; none for Tasks 01–02.
**Follow-up:** revisit before any Stage-D local prototyping.
