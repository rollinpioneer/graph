python -B -m p2crl.cli prepare --repo $WT --protocol $WT/experiments/pathgraph_p2c_rl_v1/protocol.json --out $DATA
python -B -m p2crl.cli dry-run --repo $WT --out $DATA/dry_run --data $DATA --no-gradients
python -B -m p2crl.cli freeze --repo $WT --out $DATA/preregistration --data $DATA --source-seed $PKG/contracts/source_lock.seed.json
execute-campaign is not invoked in prepare mode
