# Gate M — observation margin review (offline, zero new samples)

- Status: **GATE_M_COMPLETE_REPLAY_REPRODUCES_R2**; environments constructed 0; images generated 0.
- Replay of the frozen production `analyze_view` over saved R2 RGB-D: 912 object/view cells, 0 mismatches.
- Evidence: replay/support/margins/self-repeat/pair are PUBLIC; coverage geometry is QA_HEURISTIC.

- Tested pad_v remaining-target sideview support (layout_0/B_PENDING/obj_b): 8 pixels, margin to min_pixels 0.
- Tested pad_v remaining-target sideview support (layout_1/C_PENDING/obj_c): 8 pixels, margin to min_pixels 0.
- This does not revoke the R2 PASS.
- Untested complementary positions (layout_0 C_PENDING obj_c, layout_1 B_PENDING obj_b) cannot be inferred from R2.
- No lowering of min_pixels, no palette change, no carry-over.
- A margin of 0 does not stop the run by itself; it forces the complementary pad_v canaries first.
- Problems: none
