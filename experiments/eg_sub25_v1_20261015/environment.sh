export EG_ROOT=/home/xushijie3/work/eg_sub25_v1_20261015
export CONDA_PREFIX="/home/xushijie3/work/eg_full_repro_v1_20261015/runtime/conda"
export CUDA_HOME="/home/xushijie3/work/eg_full_repro_v1_20261015/runtime/cuda121"
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export MAX_JOBS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_CUDA_ARCH_LIST=8.0
export PIP_CACHE_DIR="$EG_ROOT/pip_cache" PIP_DISABLE_PIP_VERSION_CHECK=1
export EG_SEED=20261015
export PYTHONPATH="$EG_ROOT/runtime/EconomicGrasp:$EG_ROOT/runtime/EconomicGrasp/libs/pointnet2:${PYTHONPATH:-}"
