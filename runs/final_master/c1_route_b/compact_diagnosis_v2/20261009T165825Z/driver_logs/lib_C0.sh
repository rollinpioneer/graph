set -x
CUDA_VISIBLE_DEVICES=2 /home/xushijie3/envs/cpdisr/bin/python scripts/c1_compact_diagnosis_v2.py lib-score --run-root /home/xushijie3/work/graph_cp_disr/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z --model C0 --device cuda:0
CUDA_VISIBLE_DEVICES=2 /home/xushijie3/envs/cpdisr/bin/python scripts/c1_compact_diagnosis_v2.py lib-equiv --run-root /home/xushijie3/work/graph_cp_disr/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z --model C0 --device cuda:0
CUDA_VISIBLE_DEVICES=2 /home/xushijie3/envs/cpdisr/bin/python scripts/c1_compact_diagnosis_v2.py train-pair-accuracy --run-root /home/xushijie3/work/graph_cp_disr/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z --models C0 --device cuda:0
