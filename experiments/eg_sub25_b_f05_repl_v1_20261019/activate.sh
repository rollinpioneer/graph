source /aita_data/home/aita_xushijie/work/grasp/eg_lqr_build_confirm_v3_20261019/activate_gpu04.sh
export REPL_ROOT="$GRASP_ROOT/eg_sub25_b_f05_repl_v1_20261019"
export EG_SEED=20261015
export PYTHONPATH="$GRASP_ROOT/eg_sub25_v1_20261015/runtime/EconomicGrasp:$GRASP_ROOT/eg_sub25_v1_20261015/runtime/EconomicGrasp/libs/pointnet2:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ulimit -n 65536
