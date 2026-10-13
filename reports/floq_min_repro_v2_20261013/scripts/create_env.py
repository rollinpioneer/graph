import subprocess,json,pathlib,time
r=pathlib.Path("/home/xushijie3/work/floq_min_repro_v2_20261013")
a=["/usr/bin/uv","venv","--python","3.8.19",str(r/"envs/floq")]
x=subprocess.run(a)
(r/"receipts/venv_result.json").write_text(json.dumps({"command":a,"returncode":x.returncode,"epoch":time.time()},indent=2))
