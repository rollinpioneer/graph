# Phase C evidence reproduction

Authorized execution only: phase_c.py <independent-root>. Never re-run without a separate approval; the eight slots have been consumed after execution.

Saved-evidence verification: analyze.py <root>. This loads no model and performs no search. Requires previous G2 evidence at preregistration previous_remote_root (or explicit analysis_previous_root for a local evidence copy).

Each branch owns the initial frozen cache, full checkpoints and raw little-endian FP32 bytes. branch_ledger.jsonl accounts for every actual single-state invocation including interruptions. Partial expansions are stored separately.

Git preserves reviewed code, tables, lightweight results and complete hashes. Full raw evidence is retained on lab-xushijie3 and copied locally; artifact_manifest.json binds every archived file.
