# Stage 3 round ZIP validation

Each small-stage ZIP was generated in a temporary delivery directory, SHA256-recorded, and passed `unzip -t`; only `stage3_complete.zip` is retained in the download directory per user instruction.

| round | sha256 | unzip_test |
|---|---|---|
| stage3_1_input_adapter | 8f05358aa263e1c88fc082ff502ea16c3a754b8d384419e3438867fa1e612843 | PASS |
| stage3_2_diagnostic_suite | 6b5f3c9c6930c09ee7c8f744a9357de19aa6aa6ff14e47d1888d980e7458e8cb | PASS |
| stage3_3_baseline_runs | 80e614738e7f92ce52801beb7a6b8afd4b75df1bf40dbf7297e9583d369f33ae | PASS |
| stage3_4_misscoring_analysis | 66089e43adfe63a6c92178a227fb08189fed53fc3391c27ad85136227027cbdd | PASS |
| stage3_5_g1_decision | f93618215f585d397dfde7d7e52a590b3eff1b1b4d34a5144353d2d48b17cab8 | PASS |

Final download: `downloads/stage3_complete.zip` only.
