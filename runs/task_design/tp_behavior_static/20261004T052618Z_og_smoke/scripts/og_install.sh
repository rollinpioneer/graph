#!/bin/bash
# Smoke-test install of Isaac Sim 5.1 + OmniGibson (pinned BEHAVIOR-1K checkout) into an isolated venv. Mirrors setup.sh; no dataset.
set -u
B=/home/xushijie2/xsj2_behavior1k
V=/home/xushijie2/xsj2_og_env
export PIP_CACHE_DIR=/home/xushijie2/xsj2_og_cache/pip
export UV_CACHE_DIR=/home/xushijie2/xsj2_og_cache/uv
export OMNI_KIT_ACCEPT_EULA=YES
unset EXP_PATH CARB_APP_PATH ISAAC_PATH PYTHONPATH CONDA_PREFIX
mkdir -p $PIP_CACHE_DIR $UV_CACHE_DIR
stage() { echo "STAGE $1 $(date -u +%H:%M:%S)"; }
fail() { echo "FAILED $1 $(date -u +%H:%M:%S)"; exit 1; }
stage start
uv venv --python 3.11 --seed $V || fail venv
stage venv_done
source $V/bin/activate
python --version
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 torchcodec==0.5 --index-url https://download.pytorch.org/whl/cu128 || fail torch
stage torch_done
python -m pip install "numpy<2" setuptools wheel || fail numpy
python -m pip install -e $B/bddl3 || fail bddl
stage bddl_done
python -m pip install -e $B/OmniGibson --no-build-isolation || fail omnigibson
stage omnigibson_done
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com || fail isaacsim
stage isaacsim_done
python -m pip install --force-reinstall cffi==1.17.1 || fail cffi
python -m pip install --force-reinstall "websockets>=15.0.1" || fail websockets
ISAAC_PATH_X=$(python -c "import isaacsim, os; print(os.environ.get('ISAAC_PATH',''))" 2>/dev/null)
echo "isaac path: $ISAAC_PATH_X"
if [ -n "$ISAAC_PATH_X" ] && [ -d "$ISAAC_PATH_X/extscache" ]; then find "$ISAAC_PATH_X/extscache" -type d -name websockets -path "*/pip_prebundle/*" -exec rm -rf {} + 2>/dev/null; fi
PK=$V/lib/python3.11/site-packages/isaacsim/extscache/omni.services.pip_archive-0.16.0+107.0.3.lx64.cp311/pip_prebundle/packaging
[ -d "$PK" ] && rm -rf "$PK"
stage fixes_done
python -m pip list 2>/dev/null | grep -i -E "^(isaacsim|omnigibson|bddl|torch|numpy) "
du -sh $V /home/xushijie2/xsj2_og_cache 2>/dev/null
stage ALL_DONE
