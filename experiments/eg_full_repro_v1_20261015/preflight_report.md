# 正式训练预检查：NOT_RELEASED

- dependencies_import: PASS
- torch_cuda_forward_backward: PASS
- knn_vs_reference: PASS
- pointnet2_fps_gather_cylinder_grouping: PASS
- minkowski_cuda_convolution_backward: PASS
- official_checkpoint_load_synthetic_inference: PASS
- complete_training_dataset: BLOCKED_DATA
- complete_supervision: NOT_RUN_DATA_BLOCKED
- real_dataloader: NOT_RUN_DATA_BLOCKED
- real_model_forward_loss_backward_update: NOT_RUN_DATA_BLOCKED
- same_experiment_checkpoint_save_reload: NOT_RUN_DATA_BLOCKED
- formal_training: NOT_RUN_DATA_BLOCKED

环境工程检查不能替代真实训练数据的前向、各项 Loss、反向更新、Checkpoint 保存重载检查。当前数据门未通过；本卡正式训练不放行。
