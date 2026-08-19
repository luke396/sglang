"""State capturers: online export of model internals during serving.

Hidden-state capture (``hidden_*.py``) exports the target model's aux/last
hidden states for draft-model training. Module map:

- ``hidden_states.py``  — orchestration seam between serving and capture:
  gates (sampling window, pressure), forward/verify/finish hooks.
- ``hidden_host.py``    — host pipeline: staging rings, device twins, the
  finalize worker, the sidecar payload cache, bookkeeping.
- ``hidden_sink.py``    — export worker and the file sink.
- ``hidden_mooncake.py``— Mooncake sink: prefix-segment publishing, manifest.
- ``hidden_pack.py``    — CUDA/CPU kernels packing committed verify rows.
- ``hidden_prefix.py``  — content-addressed segment identity (hash chain).

Glossary (used throughout the hidden_* modules):

- **staging ring** — pinned-host ring buffer receiving async D2H copies of
  hidden rows; one ring for prefill, a separate one for verify.
- **twin** — capture-owned HBM buffer holding one verify step's packed rows;
  exists so the next CUDA-graph replay cannot overwrite rows before D2H.
- **sidecar** — bounded host-side payload cache keyed by KV token slot;
  the single place hidden rows live between finalize and export.
- **lease** — an export hold on exact (slot, generation) sidecar rows; a
  leased row survives KV-slot reuse until its sample settles.
- **pin** — a reader's short-lived hold on a physical sidecar row during a
  gather; writers copy-on-write around pins instead of waiting.
- **COW / generation** — sidecar rows are versioned; an odd generation marks
  an in-place write window, exact generation match validates identity (ABA).
- **finalize** — the single writer thread that drains both staging rings in
  global sequence order into the sidecar.
- **settle** — release a sample's leases after its export succeeds or is
  attributed as a miss.
- **degrade window** — a period where capture drops new work because some
  queue crossed the pressure watermark; serving never blocks on capture.
- **capture miss** — a request whose sample is dropped for any reason; every
  miss path increments an attributing ``*_miss_ct`` counter.
- **sampling window** — the periodic wall-clock window during which arriving
  requests are captured (window/period envs); decisions stick per request.
- **manifest** — the per-writer append-only discovery stream in Mooncake
  (``_seq/{dp_rank}/{n}``); consumers tail it knowing only the store prefix.
- **segment** — an immutable, content-addressed run of hidden rows in
  Mooncake, shared across samples with a common prefix.

Fail-closed philosophy: any inconsistency (identity mismatch, capacity,
pressure, shutdown) drops whole samples with an attributing counter; capture
never backpressures or corrupts serving.

``routed_experts.py`` / ``indexer_topk.py`` are unrelated per-request
capturers returned inline with responses.
"""
