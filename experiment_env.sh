#!/usr/bin/env bash
source /home/xushijie/CUPID/env.sh
cd "$REPO_DIR"
# User-authorized resource remapping on 2026-07-21: physical GPU 0 was full.
export CUDA_VISIBLE_DEVICES=1

export TRAIN_DATE="20260720_cupid_square_minrep"
export TRAIN_NAME="${TRAIN_DATE}_train_diffusion_unet_lowdim_square_mh_0"
export TRAIN_DIR="$REPO_DIR/data/outputs/train/${TRAIN_DATE}/${TRAIN_NAME}"

export EVAL_DATE="20260720_cupid_square_rollout100"
export EVAL_DIR="$REPO_DIR/data/outputs/eval_save_episodes/${EVAL_DATE}/${TRAIN_NAME}/latest"

export SMOKE_DATE="20260720_cupid_square_smoke"
export SMOKE_NAME="${SMOKE_DATE}_train_diffusion_unet_lowdim_square_mh_0"
export SMOKE_DIR="$REPO_DIR/data/outputs/train/${SMOKE_DATE}/${SMOKE_NAME}"
