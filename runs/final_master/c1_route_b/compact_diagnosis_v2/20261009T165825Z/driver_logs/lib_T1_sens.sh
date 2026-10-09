set -x
while pgrep -f "panel --run-root /home/xushijie3/work/graph_cp_disr/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z --arms T1 --set joint32" > /dev/null; do sleep 5; done
export CUDA_VISIBLE_DEVICES=6
/home/xushijie3/envs/cpdisr/bin/python scripts/c1_compact_diagnosis_v2.py lib-score --run-root /home/xushijie3/work/graph_cp_disr/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z --model T1 --device cuda:0
/home/xushijie3/envs/cpdisr/bin/python scripts/c1_compact_diagnosis_v2.py lib-equiv --run-root /home/xushijie3/work/graph_cp_disr/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z --model T1 --device cuda:0
/home/xushijie3/envs/cpdisr/bin/python scripts/c1_compact_diagnosis_v2.py train-pair-accuracy --run-root /home/xushijie3/work/graph_cp_disr/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z --models T1 --device cuda:0
for a in D0 C0 T1; do for u in 820 1640 2460 3280 4100; do /home/xushijie3/envs/cpdisr/bin/python scripts/c1_compact_diagnosis_v2.py lib-score --run-root /home/xushijie3/work/graph_cp_disr/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z --model $a@$u --device cuda:0; done; done
/home/xushijie3/envs/cpdisr/bin/python scripts/c1_compact_diagnosis_v2.py train-pair-accuracy --run-root /home/xushijie3/work/graph_cp_disr/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z --models D0@820,D0@1640,D0@2460,D0@3280,D0@4100,C0@820,C0@1640,C0@2460,C0@3280,C0@4100,T1@820,T1@1640,T1@2460,T1@3280,T1@4100 --device cuda:0
