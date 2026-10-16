# sha256 receipts: minimal checks + experiment A (2026-10-16)

Remote: /home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/exp_ab/

Note: the card says checkpoint hashes are recorded before the runs. They were recorded after experiment A,
from the unchanged files under runs/formal/ (the experiment scripts only read runs/, never write there).

## Checkpoints read (runs/formal/*/out/gfp_OGBench_Env_humanoidmaze_medium_navigate_singletask_task1_v0/)
G_task1_s0 = date_20261016_020227_seed_0 (config seed 0); G_task1_s1 = date_20261016_020257_seed_1; F_task1_s0 = date_20261016_020242_seed_0
829ebc6345c4372e17f5f8aba7ef3d6fee32700a25aa4e515a3170550a6815c9  G_task1_s0 params_500000.pkl
28ce6c95a75223675794fe5723a0d455a78efbebeb288628a8035b239285c0c7  G_task1_s0 params_600000.pkl
145541f12c34f2b5c3f0a16a0f9eb231c0d804057eb3d8481c6760495474bdce  G_task1_s0 params_700000.pkl
7ee47206f2494ae9559a1996ef82246a2df4c052e94b47cf349719244c911815  G_task1_s0 params_800000.pkl
107187aaca182a341dc20ca11f3e1366354056b4ba78a97c61fb7430a5b5e492  G_task1_s1 params_500000.pkl
03a4e26563d3d3a3891e9712bc24f64810208420e21a61500594b30174ecc204  G_task1_s1 params_600000.pkl
b0db7254fb9ae7b16f87adc72bee9e77a37152d1077195490f5eba7d653a683b  G_task1_s1 params_700000.pkl
6899296d666e85a18b6cd82963b3c7891698e89da95f6f6d90c1be8bab2db1de  G_task1_s1 params_800000.pkl
7e67a561a5ea2483e52a8e674c96d07fdc668a2c871cad7dbb13e48c85fef798  F_task1_s0 params_500000.pkl
81991bfc9ba17420fa4b3c7a5e2a53bb5f5605a57d2630e1bf512c04c69814dd  F_task1_s0 params_600000.pkl
4633853560a2f71ef2d80fa156b79cdc960781d83bc70da0e38f887e8bba5ab7  F_task1_s0 params_700000.pkl
485befc3b873cb8ece1bb8db96b1f05f83bfa50ddb10dcb1e6b7520d72aae23f  F_task1_s0 params_800000.pkl

## Outputs (exp_ab/)
52368b8b15b0130ae67bf078ef4b260cc6b3a761351332d377dddcd60aea5d0c  mincheck.json   (copied to results/exp_ab/, hash verified)
8260917f9a8234a084403a4d301d13ba3b57c7bcc3ea6ce2849378644db77629  eval_A0.json    (per-episode; summary in results/exp_ab/eval_A_summary.json)
3a712a4ce3baa732f14aea96a908bcfbeb1fe948bc756f491801556fbd4b4431  eval_A1.json
06998590505d538b28cdf8e57750f1797a9097d6d8bbd3810a868b66f19f6568  eval_A2.json
ae36c9c091d5c9b0a00650d5a59853878a55145dd9d77cfbdea67b10aecc806b  eval_A3.json
64bea71c27b4083b7c2d5368f51201b3c07e15e1b121ef44855c025c23dd16f0  actors/train_A0.json   (copied, hash verified)
e1c7ed9b79fc3fc363ec90005ff54d5cc67a657e10ce7a61ba5acc6dd62131df  actors/train_A1.json   (copied, hash verified)
54271d7be78055611438df71d08158b878c91c687d9fe1470a34bdd9390d1c2e  actors/train_A2.json   (copied, hash verified)
ecb284023c449b8db0fa64d246afb0575b9775a9456bb86df2946b8a024c5897  actors/train_A3.json   (copied, hash verified)

## Actor checkpoints (exp_ab/actors/, not copied)
1cf27446986a9a18f806d56505b81dc6584a4e6710283f523ad707d78a1c43fb  actors/actor_A0_10000.pkl
ea04a0ecdbab6bbdb68490e90013a9921ab1d724054494007dbcf2cd732b5191  actors/actor_A0_20000.pkl
2932c4e01d43e963c19ed460cc74aeda7cc1890f10cd38cbd58ff0bac82e482a  actors/actor_A1_10000.pkl
109dfa69785e5e84a2f46409c3e23c32b808603ce0f25733b9d32d450291602b  actors/actor_A1_20000.pkl
5c6f6909d638413a2ecfcaaa22d9b82722a0981be8e77f4c757951caf1e81526  actors/actor_A2_10000.pkl
4698e14b0114b3024d74636ebf9e9cf5b0ed646968767159f53c6505abe98ca4  actors/actor_A2_20000.pkl
42f58d68390f78ca3b18ce4b150836927156d4802e5f30d24b85a65b08c2eb89  actors/actor_A3_10000.pkl
c52832de5d5d40259b9d7898bc322faa92a1ee210c6f2e9cfca5ade64fe25f90  actors/actor_A3_20000.pkl

## Code (code/, remote == local)
21532739d8f1ab2c8aab2373fa4d37066591c3a94e8e82ede19dd21af04a4c2b  code/mincheck.py
3f0d74004fe000c22a39961f4bcca9ca44b04ec5eaeafb4180d81060925915af  code/exp_a.py
b0cdc490eb00b4486756e79051efd939fd91bcfad47504287b8e0c00b3bbb2f5  code/exp_a_eval.py
f7781b9b12b4e3b3858208d36c6340948accaab18338b644e3a269c167a3f0ce  code/exp_b.py
d6eaf220476edcac25e6da7cb421c8002806e8acda65c520b7a9f5a3e5811251  code/exp_b_eval.py

# Experiment B (authorized 2026-10-16 06:41 UTC via decision card; run 14:42-14:55 CST on gpu03 GPUs 0/5/7)

## Outputs (exp_ab/)
af968375757fda9637f67b7210efb459561203a3a1e70dad994325aef46103c8  B_EVAL_POINTS.txt
28b1afae707a51307465a386c7219265092dcce6d2e67a8f4b92cee795944398  b_B0/train_B0.json
0dd2fc891e266c63c2b806f3b80a29a9bc985dc80708e7e8feef418ff47aecc2  b_B1/train_B1.json
4400a386566e98b787510b0721c39061c60716408d57cca834e7ced5a3b4b767  b_B2/train_B2.json
98fe41ed28f92182db3059d39b597cd911731b6c041e068650ad240cb2b2d346  eval_B0.json
a6ed1531222da607ae8bccb7fdb33a5e2b20641642896e60ae34cbdf7a88f670  eval_B1.json
2f58d77cc8e0a58c4dc805f6b8090d158eb07c5a41e649ad9b164a29ccfb6f60  eval_B2.json
dfcc208ab47c0134dea9af27419a372e5bc548d762ae9b787387f2c767341052  eval_B_summary.json
(train_B*.json copied to results/exp_ab/b/, eval_B_summary.json and B_EVAL_POINTS.txt copied; hashes verified)

## B checkpoints (exp_ab/b_B*/, not copied; full list in results/exp_ab/b_SHA256SUMS)
B0 params_700000.pkl 0b4b297f... differs from original G_task1_s0 params_700000.pkl 145541f1... -> B0 is a paired continuation, not a bit-exact replay.
