# L2RA-R1 revision2 package bodies

The ZIP entities are retained outside Git in this directory on the execution host.
Use `package_index.json` and the adjacent SHA256 sidecars to verify them.

- Aggregate: `/home/__compress_data/xushijie/graph_l2ra_r1_worktree/downloads/l2ra_r1_v2_revision2/L2RA_R1_results.zip`
- Aggregate SHA256: `55f9e3735dcf1bc397df5ffeacc7b62b6f1ed4efa3ca0fc3cc9f6464dcfeefdf`
- Round ZIP entities are indexed rather than nested in the aggregate.
- Restore by placing each original ZIP at the absolute path recorded in `package_index.json`, then verify its external and internal SHA256 sidecars.

No checkpoint, raw RGB collection, complete dense trajectory cache, API key, or sensitive log is included in Git.
