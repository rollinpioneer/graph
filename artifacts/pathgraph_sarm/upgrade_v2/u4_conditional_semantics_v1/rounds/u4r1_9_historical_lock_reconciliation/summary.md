# U4R1.9 historical lock reconciliation

The original U4R1 lock was stale for five committed U4B metadata/manifests because the U4B completion snapshot added the existing `cupid` CPU/PyTorch execution metadata after the first lock content was prepared. The U4B rollout payloads, graph files, family lock, and checkpoint were unchanged.

The replacement lock records the exact 60 committed U4B/U4R1 input files at source snapshot `d30b4a8ce0d6e7afb1bdccd2d63d5c70268518c8`, preserves the superseded lock SHA for audit, and validates successfully. No scientific rerun, API call, checkpoint write, or training job was performed.
