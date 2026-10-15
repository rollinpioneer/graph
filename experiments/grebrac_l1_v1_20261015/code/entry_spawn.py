"""Run the official GFP main.py unchanged, with multiprocessing 'spawn' (MuJoCo EGL fork issue seen in earlier runs)."""
if __name__ == "__main__":
    import multiprocessing, runpy, sys
    from pathlib import Path
    multiprocessing.set_start_method("spawn", force=True)
    main_path = Path(sys.argv[1]).resolve()
    sys.argv = [str(main_path), *sys.argv[2:]]
    sys.path.insert(0, str(main_path.parent))
    runpy.run_path(str(main_path), run_name="__main__")
