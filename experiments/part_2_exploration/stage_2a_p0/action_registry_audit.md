# Action registry audit

- Formal Stage 2A schemas: OPEN, PICK, PLACE, PLACE_BUFFER.
- MOVE is rejected for all Stage 2A tasks and is not a PLACE_BUFFER alias.
- T_C intermediate relocation is PICK(interferer) + PLACE_BUFFER(interferer, buffer).
- Stage 0C T_A/T_C caches are historical and not reused.
