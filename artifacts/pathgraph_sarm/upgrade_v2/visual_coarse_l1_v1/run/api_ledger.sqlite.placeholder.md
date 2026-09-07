# Externalized API Attempt Ledger

- Original path: `artifacts/pathgraph_sarm/upgrade_v2/visual_coarse_l1_v1/run/api_ledger.sqlite`
- Original filename: `api_ledger.sqlite`
- Size at completion: 24576 bytes.
- SHA256: `c87aa97a74c98ae8aca2f3703047d420bc61337b48f1ab7e09d4be2e55a05cc9`.
- Purpose: authoritative shared counter for all 74 Qwen transmission attempts.
- Git/ZIP status: intentionally omitted; the exported status and usage CSV files contain no credentials.
- Restore: retain the local SQLite file or reconstruct only for audit from provider records and the committed per-request states; never reset it to obtain more calls.
