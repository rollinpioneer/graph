# Runtime perception patch scope (read from source text)

- tp_sr_runtime.py sha256: 7db3002c25f92fe014a71bd9b955c809fd079b84df43e5668148665bbcf62408
- perception.py sha256 (file bytes): 32b2ad455dbb418e13a4cd9cc16e8995b1bb5dc0f4bdd08d6ab1913c63930c7c

Relevant source lines:

```
def _nearest_palette_mask(rgb, color, tol):
def install_nearest_palette_mask():
_perception._mask = _nearest_palette_mask
install_nearest_palette_mask()
```

- Dynamic replacement target: `perception._mask` (module-level attribute assigned at runtime).
- Timing: installed when a T_P_SR runtime bundle is created, before observations are taken.
- Process scope: process-wide for the lifetime of that worker process; not limited to one branch.
- Unchanged file bytes of perception.py do not mean unchanged behaviour: the nearest-palette mask changes mask assignment for pixels claimed by several per-colour tolerance masks, and may change held/shaded interferer pixel counts (hypothesis, unverified).
- Effect on the observed relocation outcomes (e.g. NO_PLAN after PICK interferer) has not been separated from other causes.
