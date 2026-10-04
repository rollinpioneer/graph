# BEHAVIOR-1K official symbolic API: smoke test of a headless Isaac Sim / OmniGibson environment

Authorised scope: smoke test only (no dataset, no BEHAVIOR dataset terms). Result: **ENGINEERING_BLOCKED** (ISAAC_SIM_RTX_RENDERER_CRASH_ON_A100). No scientific conclusion.

1. Install succeeded in an isolated Python 3.11 venv (Isaac Sim 5.1.0.0, OmniGibson 3.9.3 editable from the pinned checkout, torch 2.7.0+cu128); one extra upstream pin was needed (warp-lang==1.12.0).
2. OmniGibson's launcher refuses to start until the robot assets are present; they come from huggingface.co, which the server cannot reach (only the third-party mirror hf-mirror.com responds), and the card scope excluded dataset downloads.
3. Isaac Sim launched directly (no OmniGibson, no assets): the kernel starts and enumerates the A100s over Vulkan, but extension libraries fail on host-OS issues (libgomp static TLS, missing libXt and libGLU on Rocky Linux 10). These were fixed in user space (torch's libgomp preloaded; libXt and mesa-libGLU RPMs extracted without root).
4. With those fixed, start-up proceeds to the RTX scene renderer and crashes (SIGSEGV, core dump) inside librtx.scenedb.plugin.so / libcarb.scenerenderer-rtx.plugin.so / libomni.hydra.rtx.plugin.so. All 8 GPUs are A100 (no RT cores); OmniGibson's documented requirement is RTX 2070+.

Evidence: smoke_results.json, crash_excerpt.txt, gpu_report_from_isaac_sim.txt, install_summary.txt, scripts/.
