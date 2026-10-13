import subprocess,json,pathlib,time
r=pathlib.Path('/home/xushijie3/work/floq_min_repro_v2_20261013')
a=json.loads((r/"receipts/install_command.json").read_text())
x=subprocess.run(a)
(r/"receipts/install_result.json").write_text(json.dumps({"returncode":x.returncode,"epoch":time.time()},indent=2))
