# sha256 receipts: candidate-c diagnostics + P0 fixed-reference (2026-10-16, no training)

Remote: /home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/

## Outputs (out/cdiag/)
df872414797a13308ce30a04cecbada615ad9f237db9658400f961893d85b718  c1_rank.json
4cbf1123851e6196afcb566c6997a9cc19a77c7f9e015e5bfdea5c0fcc7a47ac  c2_a6a.json
34ac7ab1f27ec2f7db281ab54e4ec0f55a67563627ed9c8aefcb73e7b085b7f2  c2_a6b.json
3957916fb6975d16358eccf5f2f3b3987747966db05e16eaaa22a136d7396192  c2_p1.json
f34c38e80806e4c8b027680b5e5bef959fbda9ccc6ac7783a03a553844cd7196  c2_p2.json
04cea531805c3e59cd547ec1c67dac7a0bcc39db06b09b69b10406cf4ed574c5  c2_p3.json
3a08dbe7eec388dca8f7223c1a45679c7a6329583f6486b3a21e7e670e4ec9b2  c2_p4.json
ea6609a89eeba42a2a445052eb7aeac0f279a9f0f53d4050ac014a93aa1d9db5  c2_t1.json
14c73d9041f2c909405b0d626ba9b11d160932cb0da449d70057a3160139d226  p0_fixedref.json
5c351a3882ea33ef78fe8ee835ce29bc2fa847f49b5eeaae9a8096241c87173d  target_gap.json
fa3b03a09a625a2d4e6002abf8cb2b4fa04043c52f07b29bacbde03d0768582e  c1_within.txt   (copied to results/cdiag/, hash verified)
576441867df51b6aa3a54607bfc89e3efc82f36273e076aade3201853bae9d5a  summary.txt     (copied to results/cdiag/, hash verified)

## Code (code/, remote == local)
80a58cdabccd79c0c7f898004ca3db8258e21920f69baa9e15a4c89a3f650b62  c1_rank.py
733d7ee7bd6f043b1dc4d336c6fb9b0453c4dd773d25bb91188654cddf42ca40  c2_replay_v1_used_for_card_arms.py  (A0-A5, F0-F3 runs)
85b31d1bc35ff6df7f9955c931250ff97b62b9360d05e07f6e75742306e7e50d  c2_replay.py                        (adds post-hoc A6; used for c2_a6a/c2_a6b)
e0cadbd1f7187cf2e4e00610b8933476b43b4c70594dd18f98e91d40590a60d9  p0_fixedref.py
76418808ef74e73cc2b8e0f411679c3cc35af28e2be67475cc1c29122916efc6  envpool.py
531d01c54b3907c29b4dc77bac57e2b75226c6f34a46781bc07899c994e47d90  c_summarize.py
bc1a2d3373ef49c4a5901061a85ee10d6887e79d5fbdd3d5eec429d54bcd19d0  c1_within.py   (post-hoc)
c8317e7f53b422c8e91beaa471666784fadb39faccbfb15eb5f552b98073249c  target_gap.py   (post-hoc)
95b2d85487175559cba134c15ee2e9bcc46ef256e502f137937ac52a5800458f  c2_final_x.py   (post-hoc; run inline on remote, output in results/cdiag/c2_final_x.txt)
f7d14f069ebf221cb136b55193f69bcba04973f60754389a50e727db6b7480ff  launch_cdiag.sh

Eval: task4 / task1 official envs, eval seed 4242, 3 rounds x 10 envs = 30 episodes per (model, arm). All checkpoints 1M except P0 pairs and target_gap specs listed in the report.
