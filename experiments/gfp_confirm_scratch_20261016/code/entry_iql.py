"""Stage-2 confirmation: run the official GFP main.py unchanged (as scripts/entry_spawn.py: multiprocessing 'spawn'),
after replacing agents['gfp'] by IQLScratchAgent. agent_name stays "gfp" so main.py / hyperparams.py apply exactly
the same per-env GFP hyper-parameters (humanoidmaze_medium_navigate row of gfp_params.csv) as the official runs.
usage: entry_iql.py main.py <official hydra args> +agent.expectile=0.7
"""
if __name__ == "__main__":
    import multiprocessing, runpy, sys, os
    from pathlib import Path
    multiprocessing.set_start_method("spawn", force=True)
    main_path = Path(sys.argv[1]).resolve()
    sys.argv = [str(main_path), *sys.argv[2:]]
    assert any(a.startswith("+agent.expectile=") for a in sys.argv), "expectile must be given"
    sys.path.insert(0, str(main_path.parent))
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    import agents
    from iql_gfp import IQLScratchAgent
    agents.agents["gfp"] = IQLScratchAgent
    runpy.run_path(str(main_path), run_name="__main__")
