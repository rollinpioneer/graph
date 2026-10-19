# Source from a shell on gpu04. Original shared environment is read-only reused.
source /aita_data/home/aita_xushijie/work/grasp/activate_grasp.sh sub25
export LQR_ROOT="$GRASP_ROOT/eg_lqr_build_confirm_v3_20261019"
export VIRTUAL_ENV="$LQR_ROOT/environment"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHON_LQR="$VIRTUAL_ENV/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=1
