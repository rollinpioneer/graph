# Manifest resolution log (Plan v1.1 / Method 2.1.1 / Document 3.1)

Standalone Experimental Plan SHA256 (zip copy) = `96588729F3B5649F73B6BFE4BB39C4FF1E540467FC45E7E87A1F3F48AA372EA5`.

Identity: document_version=3.1, method_version=2.1.1, training_profile_revision=2.1.1, plan_version=1.1.

## Template leftovers (source package not silently edited)

- `interfaces/runtime_manifest.template.yaml` still labels method_version=2.1 and document_version=3.0. Derived runtime uses 2.1.1 / 3.1.
- `paper_method_manifest.yaml` generic evaluation seed lists do not replace Stage-specific seeds. First pause uses seed0 only.
- Package status templates remain NOT_STARTED. They were not copied over old `experiments/stage_status/*.json` actual ledgers.

## Runtime derivation

- New runtime: `experiments/manifests/runtime_manifest_v211.yaml`.
- Old D0 `experiments/manifests/runtime_manifest.yaml` left in place.
- T_A exits the default matrix. New 2A = T_B/T_C x B0/B1-K/B2/Full x seed0.
- fewshot_manifest bound to existing frozen 3 examples; model access verified only if Stage 0C issued a live T_B request.

