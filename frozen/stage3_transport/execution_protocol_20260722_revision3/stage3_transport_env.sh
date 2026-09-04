#!/usr/bin/env bash

source /home/xushijie/CUPID/env.sh
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

export STAGE3_TRANSPORT_DATE=20260722_cupid_transport_stage3
export STAGE3_TRANSPORT_NAME="${STAGE3_TRANSPORT_DATE}_train_diffusion_unet_lowdim_transport_mh_0"
export STAGE3_TRANSPORT_TRAIN_DIR="$REPO_DIR/data/outputs/train/$STAGE3_TRANSPORT_DATE/$STAGE3_TRANSPORT_NAME"
export STAGE3_TRANSPORT_EVAL_DATE=20260722_cupid_transport_stage3_rollout100
export STAGE3_TRANSPORT_EVAL_DIR="$REPO_DIR/data/outputs/eval_save_episodes/$STAGE3_TRANSPORT_EVAL_DATE/$STAGE3_TRANSPORT_NAME/latest"
export STAGE3_TRANSPORT_DATASET="$REPO_DIR/data/robomimic/datasets/transport/mh/low_dim_abs.hdf5"
export STAGE3_TRANSPORT_TEST_START_SEED=100000
export STAGE3_TRANSPORT_NUM_ROLLOUTS=100

