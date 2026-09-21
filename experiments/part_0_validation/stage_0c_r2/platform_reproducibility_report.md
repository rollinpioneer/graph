# Clean platform reproducibility

Clean detached worktree: `/home/__compress_data/xushijie/platforms/LIBERO_cpdisr_8f1084` at `8f1084e3132a39270c3a13ebe37270a43ece2a01`; `git status --porcelain` was empty. Versions: LIBERO 0.1.1, robosuite 1.4.0, MuJoCo 3.6.0, Python 3.10.19. A fixed-seed read-only LIBERO reset/render probe used seed 17 and produced RGB 96x96x3 and depth 96x96x1 twice with identical hashes. The probe executed zero robot actions. The custom Stage 0C adapter uses the same MuJoCo rendering environment with project-owned box geometry and fixed 128x128 `agentview` camera.

The one-scene adapter success probe wrote only a temporary `/tmp` output and executed zero skills. No VLM request was sent.
